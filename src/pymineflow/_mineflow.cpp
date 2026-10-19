// pybind11 bindings for MineFlow (Deutsch et al. 2022, MIT).
// NumPy arrays are passed straight to C++: no per-block callbacks into Python.
//
//   Pit          full regular block model; precedence from a pattern (1:5, 1:9, MinSearch or the full
//                cone), block sizes, azimuth-slope sectors and per-cell slope domains
//   ExplicitPit  explicit precedences (src[k] requires dst[k]) or CSR, stored in a compact CSR graph
//
// Conventions (MineFlow's): linear index = (iz*ny + iy)*nx + ix, z grows upwards; azimuth in degrees,
// 0 = +y (north), 90 = +x (east), clockwise; slope in degrees from the horizontal, in (0, 90).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mineflow.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mvd::mineflow;

using ArrI64 = py::array_t<int64_t, py::array::c_style>;
using AzimuthSlope = std::pair<double, double>;  // (azimuth deg, slope deg)

constexpr double kTau = 6.283185307179586476925286766559;
// Subtracted from every slope (radians) so that offsets lying EXACTLY on the cone are always included.
// Without it, inclusion would depend on one ulp and on the compiler (e.g. offset (4, 0, 5) with
// 7.62 x 6.096 m blocks at 45 degrees).
constexpr double kSlopeEps = 1e-9;

static double to_radians(double degrees) { return degrees * kTau / 360.0; }

// Any integer array-like -> contiguous int64. Floats, booleans and objects raise TypeError instead of
// being truncated (lists included); uint64 beyond the int64 range raises OverflowError.
static ArrI64 as_int64(const py::handle& obj, const char* name)
{
    const py::module_ np = py::module_::import("numpy");
    py::array a = np.attr("asarray")(obj);
    const char kind = a.dtype().kind();
    if (a.size() > 0 && kind != 'i' && kind != 'u') {
        std::string msg = std::string(name) + " must be integers, got dtype " + py::str(a.dtype()).cast<std::string>();
        if (kind == 'f') msg += ": scale real values with scale_values()";
        throw py::type_error(msg);
    }
    if (kind == 'u' && a.dtype().itemsize() == 8 && a.size() > 0 &&
        np.attr("max")(a).cast<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error(std::string(name) + " exceed the int64 range");
    return np.attr("ascontiguousarray")(a, "dtype"_a = "int64").cast<ArrI64>();
}

static std::string shape_of(const py::array& a)
{
    std::string out = "(";
    for (py::ssize_t i = 0; i < a.ndim(); ++i) out += (i ? ", " : "") + std::to_string(a.shape(i));
    return out + (a.ndim() == 1 ? ",)" : ")");
}

static int64_t checked_size(int64_t nx, int64_t ny, int64_t nz)
{
    if (nx <= 0 || ny <= 0 || nz <= 0)
        throw std::invalid_argument("nx, ny and nz must be positive, got (" + std::to_string(nx) + ", " +
                                    std::to_string(ny) + ", " + std::to_string(nz) + ")");
    if (nx > std::numeric_limits<int64_t>::max() / ny / nz)
        throw std::invalid_argument("nx*ny*nz does not fit in int64");
    return nx * ny * nz;
}

// Values read directly from the NumPy buffer. PseudoSolver reads them ONCE (on construction and in
// UpdateValues) and keeps no pointer, so the array only has to live during that call.
class NumpyValues : public IBlockValues {
public:
    explicit NumpyValues(const ArrI64& v) : array_(v), data_(v.data()), n_(v.size()) {}
    IndexType NumBlocks() const override { return static_cast<IndexType>(n_); }
    void BlockValue(IndexType i, ValueType* value) const override { *value = data_[i]; }

private:
    ArrI64 array_;
    const int64_t* data_;
    int64_t n_;
};

// The solver accumulates values in int64: the sum of |values| must fit.
static void check_values(const ArrI64& v, int64_t n)
{
    if (v.ndim() != 1)
        throw std::invalid_argument("values must be a 1-D array (x fastest, then y, then z), got shape " +
                                    shape_of(v));
    if (v.size() != n)
        throw std::invalid_argument("expected " + std::to_string(n) + " values (one per cell), got " +
                                    std::to_string(v.size()));
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    const int64_t* p = v.data();
    int64_t positive = 0, negative = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t x = p[i];
        if (x > 0) {
            if (positive > kMax - x) throw std::overflow_error("sum of positive values overflows int64: use a smaller scale");
            positive += x;
        } else if (x < 0) {
            if (x == std::numeric_limits<int64_t>::min() || negative > kMax + x)
                throw std::overflow_error("sum of negative values overflows int64: use a smaller scale");
            negative -= x;
        }
    }
    if (positive > kMax - negative)
        throw std::overflow_error("sum of |values| exceeds int64: use a smaller scale (see scale_values)");
}

static std::shared_ptr<IBlockValues> wrap_values(const py::handle& values, int64_t n)
{
    const ArrI64 v = as_int64(values, "values");
    check_values(v, n);
    return std::make_shared<NumpyValues>(v);
}

// Below this slope (degrees) the cone is so flat that the pattern search would not end.
constexpr double kMinSlope = 1e-3;

static void check_slope(double slope)
{
    if (!std::isfinite(slope) || slope < kMinSlope || slope >= 90.0)
        throw std::invalid_argument("each slope must be in [0.001, 90) degrees, got " + std::to_string(slope));
}

// Constant slope or azimuth-slope sectors -> SlopeDefinition (radians), as in MineFlow's InitMinSearch.
static SlopeDefinition make_slopes(const std::optional<double>& slope_deg,
                                   const std::optional<std::vector<AzimuthSlope>>& sectors)
{
    if (slope_deg && sectors) throw std::invalid_argument("pass slope_deg or sectors, not both");
    if (!sectors) {
        const double s = slope_deg.value_or(45.0);
        check_slope(s);
        return SlopeDefinition::Constant(to_radians(s) - kSlopeEps);
    }
    if (sectors->empty()) throw std::invalid_argument("sectors must not be empty");
    std::vector<AzmSlopePair> pairs;
    pairs.reserve(sectors->size());
    for (const auto& [azimuth, slope] : *sectors) {
        if (!std::isfinite(azimuth)) throw std::invalid_argument("sector azimuth must be finite");
        check_slope(slope);
        double a = std::fmod(azimuth, 360.0);
        if (a < 0) a += 360.0;
        if (a >= 360.0) a = 0.0;  // e.g. -1e-14 + 360 rounds to 360: the same direction as 0
        pairs.emplace_back(to_radians(a), to_radians(slope) - kSlopeEps);
    }
    // Repeated azimuths (e.g. 0 and 360) make the interpolation undefined.
    std::sort(pairs.begin(), pairs.end());
    for (size_t k = 1; k < pairs.size(); ++k)
        if (pairs[k].Azimuth == pairs[k - 1].Azimuth)
            throw std::invalid_argument("repeated azimuth in sectors (modulo 360): use distinct azimuths");
    return SlopeDefinition(pairs);
}

// Reusable solver: precedences and structure are built ONCE.
class SolverBase {
public:
    explicit SolverBase(int64_t n) : n_(n) {}
    virtual ~SolverBase() = default;

    // (mask, value, n_contained, seconds); releases the GIL while solving.
    py::tuple solve()
    {
        Busy busy(this);
        return solve_unlocked();
    }

    // Replaces the values and solves again. The precedence structure is reused; the solve itself starts
    // from scratch (MineFlow's UpdateValues resets the flow).
    py::tuple solve_with_values(const py::handle& values)
    {
        Busy busy(this);
        auto vals = wrap_values(values, n_);
        solver_->UpdateValues(vals);
        return solve_unlocked();
    }

protected:
    // One solve at a time per object: the GIL is released while solving, so another Python thread could
    // otherwise enter the same solver. A failed solve leaves the solver unusable.
    class Busy {
    public:
        explicit Busy(SolverBase* owner) : owner_(owner)
        {
            if (owner_->broken_) throw std::runtime_error("a previous solve failed: build a new object");
            if (owner_->busy_.exchange(true))
                throw std::runtime_error("this object is already solving in another thread: use one object per thread");
        }
        ~Busy() { owner_->busy_ = false; }
        Busy(const Busy&) = delete;
        Busy& operator=(const Busy&) = delete;

    private:
        SolverBase* owner_;
    };

    py::tuple solve_unlocked()
    {
        PseudoSolverSolveInfo info;
        double seconds = 0.0;
        try {
            py::gil_scoped_release release;
            const auto start = std::chrono::steady_clock::now();
            solver_->Solve(&info);
            seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        } catch (...) {
            broken_ = true;
            throw;
        }
        py::array_t<bool> mask(n_);
        bool* out = mask.mutable_data();
        for (int64_t i = 0; i < n_; ++i) out[i] = solver_->InMinimumCut(i);
        return py::make_tuple(mask, static_cast<int64_t>(info.ContainedValue), info.NumContainedNodes, seconds);
    }

    std::atomic<bool> busy_{false};
    bool broken_ = false;
    int64_t n_;
    std::unique_ptr<PseudoSolver> solver_;
};

// Precedences of a subset of cells of a regular grid (e.g. the ring between two nested pits), read
// lazily from the grid's pattern(s): no arc list is built. Arcs to cells outside the subset are
// dropped, which is exact when the subset is "upper closure minus lower closure": the targets are
// then in the lower closure (already mined) and never outside the upper one.
class SubsetPrecedence;

class SubsetSource : public IBlockIndexInputIteratorSource {
public:
    SubsetSource(const SubsetPrecedence* owner, const PrecedencePattern* pattern, int64_t x, int64_t y, int64_t z)
        : owner_(owner), pattern_(pattern), x_(x), y_(y), z_(z)
    {
        advance();
    }
    IndexType Next() override
    {
        const IndexType out = next_;
        advance();
        return out;
    }
    bool HasMore() const override { return next_ >= 0; }

private:
    inline void advance();
    const SubsetPrecedence* owner_;
    const PrecedencePattern* pattern_;
    int64_t x_, y_, z_;
    size_t k_ = 0;
    IndexType next_ = -1;
};

class SubsetPrecedence : public IPrecedenceConstraints {
public:
    SubsetPrecedence(int64_t nx, int64_t ny, int64_t nz, const std::vector<PrecedencePattern>& patterns,
                     std::shared_ptr<const std::vector<IndexType>> keys, std::vector<int64_t>&& cells)
        : nx_(nx), ny_(ny), nz_(nz), patterns_(patterns), keys_(std::move(keys)), cells_(std::move(cells)),
          local_(static_cast<size_t>(nx * ny * nz), -1)
    {
        for (size_t i = 0; i < cells_.size(); ++i) {
            const int64_t c = cells_[i];
            if (c < 0 || c >= nx * ny * nz)
                throw std::invalid_argument("cell " + std::to_string(c) + " is out of the grid");
            if (local_[static_cast<size_t>(c)] >= 0) throw std::invalid_argument("cell " + std::to_string(c) + " is repeated");
            local_[static_cast<size_t>(c)] = static_cast<IndexType>(i);
        }
    }
    IndexType NumBlocks() const override { return static_cast<IndexType>(cells_.size()); }
    BlockIndexInputIteratorBase Antecedents(IndexType i) const override
    {
        const int64_t c = cells_[static_cast<size_t>(i)];
        const int64_t z = c / (nx_ * ny_), rest = c % (nx_ * ny_);
        return BlockIndexInputIteratorBase(new SubsetSource(this, &pattern_of(c), rest % nx_, rest / nx_, z));
    }
    BlockIndexInputIteratorBase Successors(IndexType) const override
    {
        throw std::logic_error("Successors is not available for subset precedences (the solver does not use it)");
    }
    IndexType NumAntecedents(IndexType i) const override { return ApproxNumAntecedents(i); }
    IndexType ApproxNumAntecedents(IndexType i) const override
    {
        return static_cast<IndexType>(pattern_of(cells_[static_cast<size_t>(i)]).Offsets.size());
    }
    IndexType NumPrecedenceConstraints() const override
    {
        return static_cast<IndexType>(cells_.size() * patterns_[0].Offsets.size());
    }

    // Local index of the cell at (x, y, z), or -1 when it is outside the grid or the subset.
    IndexType local(int64_t x, int64_t y, int64_t z) const
    {
        if (x < 0 || x >= nx_ || y < 0 || y >= ny_ || z < 0 || z >= nz_) return -1;
        return local_[static_cast<size_t>((z * ny_ + y) * nx_ + x)];
    }

private:
    const PrecedencePattern& pattern_of(int64_t cell) const
    {
        return keys_ ? patterns_[static_cast<size_t>((*keys_)[static_cast<size_t>(cell)])] : patterns_[0];
    }
    int64_t nx_, ny_, nz_;
    const std::vector<PrecedencePattern>& patterns_;
    std::shared_ptr<const std::vector<IndexType>> keys_;
    std::vector<int64_t> cells_;
    std::vector<IndexType> local_;
};

inline void SubsetSource::advance()
{
    next_ = -1;
    while (k_ < pattern_->Offsets.size()) {
        const auto& o = pattern_->Offsets[k_++];
        const IndexType j = owner_->local(x_ + o.x, y_ + o.y, z_ + o.z);
        if (j >= 0) {
            next_ = j;
            return;
        }
    }
}

class Pit : public SolverBase {
public:
    Pit(int64_t nx, int64_t ny, int64_t nz, const py::object& values, const std::string& pattern,
        std::optional<double> slope_deg, int n_benches, std::optional<std::array<double, 3>> block_size,
        std::optional<std::vector<AzimuthSlope>> sectors, const py::object& domains,
        std::optional<std::vector<std::vector<AzimuthSlope>>> domain_slopes, bool full_pattern)
        : SolverBase(checked_size(nx, ny, nz)), nx_(nx), ny_(ny), nz_(nz)
    {
        auto vals = wrap_values(values, n_);
        const std::array<double, 3> size = block_size.value_or(std::array<double, 3>{1.0, 1.0, 1.0});
        for (double s : size)
            if (!std::isfinite(s) || s <= 0.0)
                throw std::invalid_argument("block_size must be three positive numbers (dx, dy, dz), got " +
                                            std::to_string(s));
        BlockDefinition grid(nx, ny, nz, 0.0, 0.0, 0.0, size[0], size[1], size[2]);
        const bool uses_domains = !domains.is_none() || domain_slopes.has_value();
        std::shared_ptr<IPrecedenceConstraints> precedence;

        if (pattern == "1:5" || pattern == "1:9") {
            // Block-count patterns: independent of block size and slope.
            if (slope_deg || block_size || sectors || uses_domains || full_pattern)
                throw std::invalid_argument("patterns '1:5' and '1:9' take no slope_deg, block_size, sectors, "
                                            "domains or full_pattern; use pattern='minsearch'");
            patterns_.push_back(pattern == "1:5" ? PrecedencePattern::OneFive() : PrecedencePattern::OneNine());
            precedence = std::make_shared<Regular3DBlockModelPatternPrecedence>(grid, patterns_[0]);
        } else if (pattern == "minsearch") {
            if (n_benches < 1) throw std::invalid_argument("n_benches must be >= 1, got " + std::to_string(n_benches));
            check_window(grid, slope_deg, sectors, domain_slopes, n_benches);
            // full_pattern: every cone offset up to n_benches (Naive: exact cone per block);
            // otherwise MineFlow's minimum pattern (same closure for a single slope definition).
            auto build = [&](const SlopeDefinition& s) {
                return full_pattern ? PrecedencePattern::Naive(grid, s, n_benches)
                                    : PrecedencePattern::MinSearch(grid, s, n_benches);
            };
            if (!uses_domains) {
                patterns_.push_back(build(make_slopes(slope_deg, sectors)));
                precedence = std::make_shared<Regular3DBlockModelPatternPrecedence>(grid, patterns_[0]);
            } else {
                precedence = domain_precedence(grid, domains, domain_slopes, slope_deg, sectors, build);
            }
        } else {
            throw std::invalid_argument("pattern must be '1:5', '1:9' or 'minsearch', got '" + pattern + "'");
        }
        solver_ = std::make_unique<PseudoSolver>(precedence, vals);
    }

    // (k, 3) int64 offsets of one domain's pattern: block (x, y, z) requires (x+dx, y+dy, z+dz).
    py::array_t<int64_t> offsets(int64_t domain) const
    {
        if (domain < 0 || domain >= static_cast<int64_t>(patterns_.size()))
            throw py::index_error("domain " + std::to_string(domain) + " out of range [0, " +
                                  std::to_string(patterns_.size()) + ")");
        const auto& p = patterns_[static_cast<size_t>(domain)];
        const py::ssize_t k = static_cast<py::ssize_t>(p.Offsets.size());
        py::array_t<int64_t> out({k, static_cast<py::ssize_t>(3)});
        auto r = out.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < k; ++i) {
            const auto& o = p.Offsets[static_cast<size_t>(i)];
            r(i, 0) = o.x;
            r(i, 1) = o.y;
            r(i, 2) = o.z;
        }
        return out;
    }

    int64_t n_domains() const { return static_cast<int64_t>(patterns_.size()); }

    // Ultimate pit restricted to `cells` (values per cell of the subset), with this grid's precedences.
    // Arcs leaving the subset are dropped: pass the cells of an upper pit that are not in a lower one.
    // Each call builds its own solver and only reads the patterns (never changed after construction), so
    // several threads may solve disjoint subsets at once (the GIL is released while solving).
    py::tuple solve_subset(const py::object& cells_obj, const py::object& values_obj)
    {
        const ArrI64 cells = as_int64(cells_obj, "cells");
        const ArrI64 values = as_int64(values_obj, "values");
        if (cells.ndim() != 1) throw std::invalid_argument("cells must be a 1-D array, got shape " + shape_of(cells));
        const int64_t m = cells.size();
        if (m == 0) return py::make_tuple(py::array_t<bool>(0), int64_t{0}, int64_t{0}, 0.0);
        check_values(values, m);
        std::vector<int64_t> subset(cells.data(), cells.data() + m);
        auto block_values = std::make_shared<NumpyValues>(values);  // holds the array: built with the GIL
        py::array_t<bool> mask(m);
        bool* out = mask.mutable_data();
        std::unique_ptr<PseudoSolver> solver;
        PseudoSolverSolveInfo info;
        double seconds = 0.0;
        {
            // Everything heavy without the GIL: building the subset, the solver and the solve, and reading the cut.
            py::gil_scoped_release release;
            const auto start = std::chrono::steady_clock::now();
            auto precedence = std::make_shared<SubsetPrecedence>(nx_, ny_, nz_, patterns_, keys_, std::move(subset));
            solver = std::make_unique<PseudoSolver>(precedence, block_values);
            solver->Solve(&info);
            for (int64_t i = 0; i < m; ++i) out[i] = solver->InMinimumCut(i);
            seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        solver.reset();  // releases the values array with the GIL held
        return py::make_tuple(mask, static_cast<int64_t>(info.ContainedValue), info.NumContainedNodes, seconds);
    }

private:
    // Cheap guard before MinSearch allocates its (2cx+1)(2cy+1)(n_benches+1)-byte search window.
    static void check_window(const BlockDefinition& grid, const std::optional<double>& slope_deg,
                             const std::optional<std::vector<AzimuthSlope>>& sectors,
                             const std::optional<std::vector<std::vector<AzimuthSlope>>>& domain_slopes, int n_benches)
    {
        double flattest = slope_deg.value_or(45.0);
        auto scan = [&](const std::vector<AzimuthSlope>& v) {
            for (const auto& sector : v) flattest = std::min(flattest, sector.second);
        };
        if (sectors) scan(*sectors);
        if (domain_slopes)
            for (const auto& v : *domain_slopes) scan(v);
        if (!(flattest >= kMinSlope && flattest < 90.0)) return;  // check_slope reports it
        const double radius = n_benches * grid.SizeZ / std::tan(to_radians(flattest));
        const double window = (2 * std::ceil(radius / grid.SizeX) + 1) *
                              (2 * std::ceil(radius / grid.SizeY) + 1) * (n_benches + 1.0);
        if (window > 2e9)
            throw std::invalid_argument("n_benches=" + std::to_string(n_benches) + " with a flattest slope of " +
                                        std::to_string(flattest) +
                                        " degrees needs a pattern search window over 2 GB: use a smaller n_benches");
    }

    template <typename Build>
    std::shared_ptr<IPrecedenceConstraints> domain_precedence(
        const BlockDefinition& grid, const py::object& domains_obj,
        const std::optional<std::vector<std::vector<AzimuthSlope>>>& domain_slopes,
        const std::optional<double>& slope_deg, const std::optional<std::vector<AzimuthSlope>>& sectors,
        Build build)
    {
        if (domains_obj.is_none() || !domain_slopes)
            throw std::invalid_argument("domains and domain_slopes go together");
        if (slope_deg || sectors)
            throw std::invalid_argument("with domains pass only domain_slopes (not slope_deg or sectors)");
        if (domain_slopes->empty()) throw std::invalid_argument("domain_slopes must not be empty");
        const ArrI64 d = as_int64(domains_obj, "domains");
        if (d.ndim() != 1 || d.size() != n_)
            throw std::invalid_argument("domains must be a 1-D array with one domain per cell, shape (" +
                                        std::to_string(n_) + ",), got shape " + shape_of(d));
        const int64_t k = static_cast<int64_t>(domain_slopes->size());
        auto keys = std::make_shared<std::vector<IndexType>>(static_cast<size_t>(n_));
        keys_ = keys;
        const int64_t* p = d.data();
        for (int64_t i = 0; i < n_; ++i) {
            if (p[i] < 0 || p[i] >= k)
                throw std::invalid_argument("domain " + std::to_string(p[i]) + " out of range [0, " +
                                            std::to_string(k) + "): one entry of domain_slopes per domain");
            (*keys)[static_cast<size_t>(i)] = p[i];
        }
        for (const auto& slopes : *domain_slopes) patterns_.push_back(build(make_slopes(std::nullopt, slopes)));
        // The solver only uses Antecedents (the requiring block's pattern). Successors of this class
        // uses the required block's pattern, but PseudoSolver never calls it.
        return std::make_shared<Regular3DBlockModelKeyedPatternsPrecedence>(grid, patterns_, keys);
    }

    int64_t nx_, ny_, nz_;
    std::vector<PrecedencePattern> patterns_;  // one per domain (or a single one)
    std::shared_ptr<const std::vector<IndexType>> keys_;  // domain of each cell (nullptr: one pattern)
};

// Index source over a contiguous slice of the CSR array (no copies).
template <typename T>
class SliceSource : public IBlockIndexInputIteratorSource {
public:
    SliceSource(const T* p, size_t n) : p_(p), n_(n) {}
    IndexType Next() override
    {
        --n_;
        return static_cast<IndexType>(*p_++);
    }
    bool HasMore() const override { return n_ > 0; }

private:
    const T* p_;
    size_t n_;
};

// Explicit precedences in CSR form: antecedents of block i = indices[indptr[i]:indptr[i+1]].
// Replaces MineFlow's ExplicitPrecedence (unordered_map + one vector per block): less memory, O(m) load.
template <typename T>
class CsrPrecedence : public IPrecedenceConstraints {
public:
    CsrPrecedence(std::vector<int64_t>&& indptr, std::vector<T>&& indices)
        : indptr_(std::move(indptr)), indices_(std::move(indices))
    {
    }
    IndexType NumBlocks() const override { return static_cast<IndexType>(indptr_.size()) - 1; }
    BlockIndexInputIteratorBase Antecedents(IndexType i) const override
    {
        const int64_t a = indptr_[static_cast<size_t>(i)], b = indptr_[static_cast<size_t>(i) + 1];
        if (a == b) return BlockIndexInputIteratorBase(nullptr);
        return BlockIndexInputIteratorBase(new SliceSource<T>(indices_.data() + a, static_cast<size_t>(b - a)));
    }
    BlockIndexInputIteratorBase Successors(IndexType) const override
    {
        throw std::logic_error("Successors is not available for CSR precedences (the solver does not use it)");
    }
    IndexType NumAntecedents(IndexType i) const override { return ApproxNumAntecedents(i); }
    IndexType ApproxNumAntecedents(IndexType i) const override
    {
        return indptr_[static_cast<size_t>(i) + 1] - indptr_[static_cast<size_t>(i)];
    }
    IndexType NumPrecedenceConstraints() const override { return static_cast<IndexType>(indices_.size()); }

private:
    std::vector<int64_t> indptr_;
    std::vector<T> indices_;
};

// (src, dst) -> CSR by counting sort, O(n + m). Runs with the GIL held: the arrays may be the caller's own,
// and another thread must not change them between the range check and the fill.
template <typename T>
static std::shared_ptr<IPrecedenceConstraints> csr_from_arcs(int64_t n, const int64_t* src, const int64_t* dst,
                                                             int64_t m)
{
    std::vector<int64_t> indptr(static_cast<size_t>(n) + 1, 0);
    for (int64_t k = 0; k < m; ++k) {
        if (src[k] < 0 || src[k] >= n || dst[k] < 0 || dst[k] >= n)
            throw std::invalid_argument("arc " + std::to_string(k) + " (" + std::to_string(src[k]) + " -> " +
                                        std::to_string(dst[k]) + ") is out of range [0, " + std::to_string(n) + ")");
        ++indptr[static_cast<size_t>(src[k]) + 1];
    }
    for (int64_t i = 0; i < n; ++i) indptr[i + 1] += indptr[i];
    std::vector<T> indices(static_cast<size_t>(m));
    std::vector<int64_t> next(indptr.begin(), indptr.end() - 1);
    for (int64_t k = 0; k < m; ++k)
        indices[static_cast<size_t>(next[static_cast<size_t>(src[k])]++)] = static_cast<T>(dst[k]);
    return std::make_shared<CsrPrecedence<T>>(std::move(indptr), std::move(indices));
}

template <typename T>
static std::shared_ptr<IPrecedenceConstraints> csr_copy(int64_t n, const int64_t* indptr, const int64_t* indices,
                                                        int64_t m)
{
    if (indptr[0] != 0 || indptr[n] != m)
        throw std::invalid_argument("indptr must start at 0 and end at len(indices) = " + std::to_string(m) +
                                    ", got " + std::to_string(indptr[0]) + " and " + std::to_string(indptr[n]));
    for (int64_t i = 0; i < n; ++i)
        if (indptr[i + 1] < indptr[i])
            throw std::invalid_argument("indptr must be non-decreasing (fails at position " + std::to_string(i + 1) + ")");
    std::vector<T> copy(static_cast<size_t>(m));
    for (int64_t k = 0; k < m; ++k) {
        if (indices[k] < 0 || indices[k] >= n)
            throw std::invalid_argument("indices[" + std::to_string(k) + "] = " + std::to_string(indices[k]) +
                                        " is out of range [0, " + std::to_string(n) + ")");
        copy[static_cast<size_t>(k)] = static_cast<T>(indices[k]);
    }
    return std::make_shared<CsrPrecedence<T>>(std::vector<int64_t>(indptr, indptr + n + 1), std::move(copy));
}

class ExplicitPit : public SolverBase {
public:
    // Block src[k] requires block dst[k] to be mined first.
    ExplicitPit(int64_t n, const py::object& src_obj, const py::object& dst_obj, const py::object& values)
        : SolverBase(n)
    {
        if (n <= 0) throw std::invalid_argument("n must be positive, got " + std::to_string(n));
        const ArrI64 src = as_int64(src_obj, "src"), dst = as_int64(dst_obj, "dst");
        if (src.ndim() != 1 || dst.ndim() != 1 || src.size() != dst.size())
            throw std::invalid_argument("src and dst must be 1-D arrays of the same length, got shapes " +
                                        shape_of(src) + " and " + shape_of(dst));
        auto vals = wrap_values(values, n_);
        auto precedence = n <= std::numeric_limits<int32_t>::max()
                              ? csr_from_arcs<int32_t>(n, src.data(), dst.data(), src.size())
                              : csr_from_arcs<int64_t>(n, src.data(), dst.data(), src.size());
        solver_ = std::make_unique<PseudoSolver>(precedence, vals);
    }

    // CSR form: block i requires indices[indptr[i]:indptr[i+1]] (e.g. a scipy.sparse.csr_matrix).
    static std::unique_ptr<ExplicitPit> from_csr(const py::object& indptr_obj, const py::object& indices_obj,
                                                 const py::object& values)
    {
        const ArrI64 indptr = as_int64(indptr_obj, "indptr"), indices = as_int64(indices_obj, "indices");
        if (indptr.ndim() != 1 || indices.ndim() != 1 || indptr.size() < 2)
            throw std::invalid_argument("indptr and indices must be 1-D arrays with len(indptr) = n + 1 >= 2");
        const int64_t n = indptr.size() - 1;
        auto pit = std::unique_ptr<ExplicitPit>(new ExplicitPit(n));
        auto vals = wrap_values(values, n);
        auto precedence = n <= std::numeric_limits<int32_t>::max()
                              ? csr_copy<int32_t>(n, indptr.data(), indices.data(), indices.size())
                              : csr_copy<int64_t>(n, indptr.data(), indices.data(), indices.size());
        pit->solver_ = std::make_unique<PseudoSolver>(precedence, vals);
        return pit;
    }

private:
    explicit ExplicitPit(int64_t n) : SolverBase(n) {}
};

// Test-only reference: solves with MineFlow's original ExplicitPrecedence (which ExplicitPit replaced
// with CSR) to check equivalence on small models.
static py::tuple solve_explicit_precedence(int64_t n, const py::object& src_obj, const py::object& dst_obj,
                                           const py::object& values)
{
    const ArrI64 src = as_int64(src_obj, "src"), dst = as_int64(dst_obj, "dst");
    if (n <= 0 || src.ndim() != 1 || dst.ndim() != 1 || src.size() != dst.size())
        throw std::invalid_argument("invalid arguments");
    auto vals = wrap_values(values, n);
    auto precedence = std::make_shared<ExplicitPrecedence>(n);
    const int64_t* s = src.data();
    const int64_t* d = dst.data();
    for (int64_t k = 0; k < src.size(); ++k) {
        if (s[k] < 0 || s[k] >= n || d[k] < 0 || d[k] >= n) throw std::invalid_argument("arc out of range");
        precedence->AddPrecedenceConstraint(s[k], d[k]);
    }
    PseudoSolver solver(precedence, vals);
    PseudoSolverSolveInfo info;
    solver.Solve(&info);
    py::array_t<bool> mask(n);
    bool* out = mask.mutable_data();
    for (int64_t i = 0; i < n; ++i) out[i] = solver.InMinimumCut(i);
    return py::make_tuple(mask, static_cast<int64_t>(info.ContainedValue));
}

// Ultimate pit OES kernels (_oes.cpp).
void register_oes(py::module_& m);

static const char* kPitDoc = R"(Ultimate pit of a full regular nx*ny*nz block model with pattern precedence.

values: 1-D int64 of length nx*ny*nz, index (iz*ny + iy)*nx + ix, z upwards.
pattern: '1:5', '1:9' (block-count patterns) or 'minsearch' (MineFlow's minimum pattern).
slope_deg: constant slope in (0, 90) degrees from the horizontal (default 45); 'minsearch' only.
n_benches: benches covered by the 'minsearch' pattern ('1:5' and '1:9' cover one bench).
block_size: (dx, dy, dz), default (1, 1, 1); 'minsearch' only.
sectors: [(azimuth, slope), ...] in degrees; azimuth 0 = +y (north), 90 = +x (east), clockwise;
         linear interpolation between sectors. Excludes slope_deg.
domains: int per cell (index into domain_slopes); domain_slopes: one [(azimuth, slope), ...] list
         per domain. Excludes slope_deg and sectors.
full_pattern: True uses every cone offset up to n_benches (exact cone per block, more arcs);
              False uses the minimum pattern (MinSearch).
Offsets lying exactly on the cone are included.
)";

PYBIND11_MODULE(_mineflow, m)
{
    m.doc() = "pybind11 bindings for MineFlow (pseudoflow ultimate pit solver, Deutsch et al. 2022)";
#if defined(_MSC_FULL_VER)
    m.attr("compiler") = "MSVC " + std::to_string(_MSC_FULL_VER);
#elif defined(__clang__)
    m.attr("compiler") = std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    m.attr("compiler") = std::string("gcc ") + __VERSION__;
#else
    m.attr("compiler") = "unknown";
#endif
    py::class_<Pit>(m, "Pit", kPitDoc)
        .def(py::init<int64_t, int64_t, int64_t, py::object, std::string, std::optional<double>, int,
                      std::optional<std::array<double, 3>>, std::optional<std::vector<AzimuthSlope>>, py::object,
                      std::optional<std::vector<std::vector<AzimuthSlope>>>, bool>(),
             py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("values"), py::kw_only(),
             py::arg("pattern") = "1:5", py::arg("slope_deg") = py::none(), py::arg("n_benches") = 8,
             py::arg("block_size") = py::none(), py::arg("sectors") = py::none(), py::arg("domains") = py::none(),
             py::arg("domain_slopes") = py::none(), py::arg("full_pattern") = false)
        .def("solve", [](Pit& self) { return self.solve(); },
             "Solve: (bool mask per cell, int64 value, n_contained, seconds). Releases the GIL; one solve at a "
             "time per object.")
        .def("solve_with_values", [](Pit& self, const py::object& values) { return self.solve_with_values(values); },
             py::arg("values"), "Replace the values and solve again, reusing the precedence structure.")
        .def("offsets", [](const Pit& self, int64_t domain) { return self.offsets(domain); }, py::arg("domain") = 0,
             "(k, 3) int64 offsets of a domain's precedence pattern (0 when there are no domains).")
        .def_property_readonly("n_domains", [](const Pit& self) { return self.n_domains(); })
        .def("solve_subset",
             [](Pit& self, const py::object& cells, const py::object& values) { return self.solve_subset(cells, values); },
             py::arg("cells"), py::arg("values"),
             "Ultimate pit restricted to a subset of cells, with this grid's precedences: (mask over the subset, "
             "value, n_contained, seconds). Arcs leaving the subset are dropped, so pass the cells of an upper pit "
             "that are not in a lower one (nested pits). Releases the GIL.");
    py::class_<ExplicitPit>(m, "ExplicitPit", "Ultimate pit with explicit precedences: src[k] requires dst[k].")
        .def(py::init<int64_t, py::object, py::object, py::object>(), py::arg("n"), py::arg("src"), py::arg("dst"),
             py::arg("values"))
        .def_static("from_csr", &ExplicitPit::from_csr, py::arg("indptr"), py::arg("indices"), py::arg("values"),
                    "CSR precedences: block i requires indices[indptr[i]:indptr[i+1]].")
        .def("solve", [](ExplicitPit& self) { return self.solve(); },
             "Solve: (bool mask, int64 value, n_contained, seconds). Releases the GIL; one solve at a time per object.")
        .def("solve_with_values",
             [](ExplicitPit& self, const py::object& values) { return self.solve_with_values(values); },
             py::arg("values"), "Replace the values and solve again, reusing the precedence structure.");
    m.def("_solve_explicit_precedence", &solve_explicit_precedence, py::arg("n"), py::arg("src"), py::arg("dst"),
          py::arg("values"), "Test-only: (mask, value) with MineFlow's original ExplicitPrecedence.");
    register_oes(m);
}
