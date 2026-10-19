// Ultimate pit OES (optimal extraction sequence) kernels, registered into the _mineflow module.
//
//   _oes_lookahead  value of every block plus the discounted value of the blocks in its downward cone
//   _oes_order      bench order: phase by phase, bench by bench from the top, and inside every phase-bench
//                   a greedy walk that maximises key - c * distance to the last block
//   _oes_order_free precedence order: phase by phase, always the available block (predecessors mined)
//                   that maximises key - c * distance, whatever its bench (the lookahead search)
//   _oes_merge_small isolated small groups of a phase move to the phase that first needs them (continuity)
//   _oes_schedule   period-by-period fill of a sequence under capacities (blocks split between periods)
//   _oes_check      independent precedence check of a sequence against the grid's precedence pattern
//
// They are called by pymineflow.oes, which validates the inputs. The pseudoflow solver is not used here.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrI = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
constexpr double kInf = std::numeric_limits<double>::infinity();

void check_size(const py::array& a, int64_t n, const char* name)
{
    if (a.ndim() != 1 || a.size() != n)
        throw std::invalid_argument(std::string(name) + " must be a 1-D array of length " + std::to_string(n));
}

// Runs f(begin, end) over [0, n) in `threads` chunks (no Python calls inside f).
template <typename F>
void parallel_for(int64_t n, int threads, F f)
{
    if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    if (threads == 1 || n < 4096) {
        f(int64_t{0}, n);
        return;
    }
    threads = static_cast<int>(std::min<int64_t>(threads, n / 1024 + 1));
    std::vector<std::thread> pool;
    const int64_t chunk = (n + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        const int64_t a = t * chunk, b = std::min(n, a + chunk);
        if (a < b) pool.emplace_back(f, a, b);
    }
    for (auto& th : pool) th.join();
}

// ---------------------------------------------------------------------------------------------------------------
// Lookahead values: LV(b) = v(b) + sum_k w[k] * (sum of v over the blocks k benches below b whose centres lie
// within radius[k] of b's centre, boundary included). Row prefix sums over the blocks' bounding box make each
// disk sum cost one subtraction per row of the disk.
py::array_t<double> lookahead(const ArrI& ix, const ArrI& iy, const ArrI& iz, const ArrD& value, double dx, double dy,
                              const std::vector<double>& radii, const std::vector<double>& weights, int threads)
{
    const int64_t n = value.size();
    check_size(ix, n, "ix");
    check_size(iy, n, "iy");
    check_size(iz, n, "iz");
    if (radii.size() != weights.size()) throw std::invalid_argument("radii and weights must have the same length");
    if (!(dx > 0) || !(dy > 0)) throw std::invalid_argument("dx and dy must be positive");
    py::array_t<double> out(n);
    if (n == 0) return out;
    const int64_t *px = ix.data(), *py_ = iy.data(), *pz = iz.data();
    const double* pv = value.data();
    const auto [x0, x1] = std::minmax_element(px, px + n);
    const auto [y0, y1] = std::minmax_element(py_, py_ + n);
    const auto [z0, z1] = std::minmax_element(pz, pz + n);
    const int64_t bx = *x0, by = *y0, bz = *z0;
    const int64_t nx = *x1 - bx + 1, ny = *y1 - by + 1, nz = *z1 - bz + 1;
    const int64_t row = nx + 1;
    const double cells = static_cast<double>(row) * static_cast<double>(ny) * static_cast<double>(nz);
    if (cells > 4e9) throw std::invalid_argument("the blocks' bounding box is too large for the lookahead (> 4e9 cells)");
    const int K = static_cast<int>(radii.size());
    // half-width (in cells along x) of the disk of bench k at row offset dy_cells, for every k
    std::vector<int64_t> reach(static_cast<size_t>(K));
    std::vector<std::vector<int64_t>> half(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        const double r = radii[static_cast<size_t>(k)];
        if (!(r >= 0) || !std::isfinite(r)) throw std::invalid_argument("radii must be finite and >= 0");
        const double tol = 1e-9 * std::max(1.0, r);
        const int64_t ry = static_cast<int64_t>(std::floor((r + tol) / dy));
        reach[static_cast<size_t>(k)] = ry;
        auto& h = half[static_cast<size_t>(k)];
        h.resize(static_cast<size_t>(2 * ry + 1));
        for (int64_t j = -ry; j <= ry; ++j) {
            const double rest = r * r - (j * dy) * (j * dy);
            h[static_cast<size_t>(j + ry)] = static_cast<int64_t>(std::floor((std::sqrt(std::max(0.0, rest)) + tol) / dx));
        }
    }
    std::vector<double> prefix(static_cast<size_t>(cells), 0.0);
    for (int64_t i = 0; i < n; ++i)
        prefix[static_cast<size_t>(((pz[i] - bz) * ny + (py_[i] - by)) * row + (px[i] - bx) + 1)] += pv[i];
    {
        py::gil_scoped_release release;
        for (int64_t r = 0; r < ny * nz; ++r) {
            double* p = prefix.data() + r * row;
            for (int64_t x = 1; x < row; ++x) p[x] += p[x - 1];
        }
        double* o = out.mutable_data();
        parallel_for(n, threads, [&](int64_t a, int64_t b) {
            for (int64_t i = a; i < b; ++i) {
                const int64_t x = px[i] - bx, y = py_[i] - by, z = pz[i] - bz;
                double s = pv[i];
                for (int k = 0; k < K && z - k - 1 >= 0; ++k) {
                    const int64_t zz = z - k - 1, ry = reach[static_cast<size_t>(k)];
                    const auto& h = half[static_cast<size_t>(k)];
                    double acc = 0.0;
                    for (int64_t j = std::max(-ry, -y); j <= std::min(ry, ny - 1 - y); ++j) {
                        const int64_t w = h[static_cast<size_t>(j + ry)];
                        const int64_t lo = std::max<int64_t>(0, x - w), hi = std::min<int64_t>(nx - 1, x + w);
                        const double* p = prefix.data() + (zz * ny + y + j) * row;
                        acc += p[hi + 1] - p[lo];
                    }
                    s += weights[static_cast<size_t>(k)] * acc;
                }
                o[i] = s;
            }
        });
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Proximity search: a set of points (block centres) with keys; the query returns the point that maximises
// key - c * distance(A, point) among the active ones, ties to the lowest id. Exact, with an implicit k-d tree
// whose nodes keep the subtree bounding box and the highest active key (branch and bound on
// key - c * distance to the box). Points are activated (available) and removed (mined) in O(log n).
class Search {
public:
    void build(const std::vector<std::array<double, 3>>& p, const std::vector<double>& key,
               const std::vector<int64_t>& id, bool active)
    {
        p_ = &p;
        key_ = &key;
        id_ = &id;
        const int m = static_cast<int>(p.size());
        m_ = m;
        perm_.resize(static_cast<size_t>(m));
        std::iota(perm_.begin(), perm_.end(), 0);
        lo_.assign(static_cast<size_t>(m), {0.0, 0.0, 0.0});
        hi_.assign(static_cast<size_t>(m), {0.0, 0.0, 0.0});
        best_key_.assign(static_cast<size_t>(m), -kInf);
        alive_.assign(static_cast<size_t>(m), active ? 1 : 0);
        pos_.assign(static_cast<size_t>(m), 0);
        build(0, m, 0);
        for (int q = 0; q < m; ++q) pos_[static_cast<size_t>(perm_[static_cast<size_t>(q)])] = q;
    }

    void activate(int point) { set(pos_[static_cast<size_t>(point)], 1); }
    void remove(int point) { set(pos_[static_cast<size_t>(point)], 0); }

    // The active point of highest key - c * distance to a (c = 0: highest key), or -1 when none is active.
    int query(const std::array<double, 3>& a, double c)
    {
        a_ = a;
        c_ = c;
        best_ = -kInf;
        best_point_ = -1;
        search(0, m_);
        return best_point_;
    }

private:
    void build(int lo, int hi, int axis)
    {
        if (lo >= hi) return;
        const int mid = (lo + hi) / 2;
        const auto& p = *p_;
        std::nth_element(perm_.begin() + lo, perm_.begin() + mid, perm_.begin() + hi, [&](int a, int b) {
            const double ca = p[static_cast<size_t>(a)][static_cast<size_t>(axis)];
            const double cb = p[static_cast<size_t>(b)][static_cast<size_t>(axis)];
            return ca < cb || (ca == cb && a < b);
        });
        build(lo, mid, (axis + 1) % 3);
        build(mid + 1, hi, (axis + 1) % 3);
        const int q = perm_[static_cast<size_t>(mid)];
        auto box_lo = p[static_cast<size_t>(q)], box_hi = box_lo;
        double k = alive_[static_cast<size_t>(mid)] ? (*key_)[static_cast<size_t>(q)] : -kInf;
        for (int child : {lo < mid ? (lo + mid) / 2 : -1, mid + 1 < hi ? (mid + 1 + hi) / 2 : -1}) {
            if (child < 0) continue;
            const auto c = static_cast<size_t>(child);
            for (size_t d = 0; d < 3; ++d) {
                box_lo[d] = std::min(box_lo[d], lo_[c][d]);
                box_hi[d] = std::max(box_hi[d], hi_[c][d]);
            }
            k = std::max(k, best_key_[c]);
        }
        lo_[static_cast<size_t>(mid)] = box_lo;
        hi_[static_cast<size_t>(mid)] = box_hi;
        best_key_[static_cast<size_t>(mid)] = k;
    }

    void set(int pos, char alive)
    {
        alive_[static_cast<size_t>(pos)] = alive;
        path_.clear();
        int lo = 0, hi = m_;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            path_.emplace_back(lo, hi);
            if (pos == mid) break;
            if (pos < mid)
                hi = mid;
            else
                lo = mid + 1;
        }
        for (auto it = path_.rbegin(); it != path_.rend(); ++it) {
            const int a = it->first, b = it->second, mid = (a + b) / 2;
            double k = alive_[static_cast<size_t>(mid)] ? (*key_)[static_cast<size_t>(perm_[static_cast<size_t>(mid)])]
                                                         : -kInf;
            if (a < mid) k = std::max(k, best_key_[static_cast<size_t>((a + mid) / 2)]);
            if (mid + 1 < b) k = std::max(k, best_key_[static_cast<size_t>((mid + 1 + b) / 2)]);
            best_key_[static_cast<size_t>(mid)] = k;
        }
    }

    double bound(int node) const
    {
        const auto s = static_cast<size_t>(node);
        if (c_ == 0.0) return best_key_[s];
        double d2 = 0.0;
        for (size_t d = 0; d < 3; ++d) {
            const double g = std::max({0.0, lo_[s][d] - a_[d], a_[d] - hi_[s][d]});
            d2 += g * g;
        }
        return best_key_[s] - c_ * std::sqrt(d2);
    }

    void search(int lo, int hi)
    {
        if (lo >= hi) return;
        const int mid = (lo + hi) / 2;
        if (best_key_[static_cast<size_t>(mid)] == -kInf || bound(mid) < best_) return;
        if (alive_[static_cast<size_t>(mid)]) {
            const int q = perm_[static_cast<size_t>(mid)];
            const auto& p = (*p_)[static_cast<size_t>(q)];
            double s = (*key_)[static_cast<size_t>(q)];
            if (c_ != 0.0) {
                const double ddx = p[0] - a_[0], ddy = p[1] - a_[1], ddz = p[2] - a_[2];
                s -= c_ * std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
            }
            if (s > best_ || (s == best_ && (*id_)[static_cast<size_t>(q)] < (*id_)[static_cast<size_t>(best_point_)])) {
                best_ = s;
                best_point_ = q;
            }
        }
        const int left = lo < mid ? (lo + mid) / 2 : -1, right = mid + 1 < hi ? (mid + 1 + hi) / 2 : -1;
        const double bl = left >= 0 ? bound(left) : -kInf, br = right >= 0 ? bound(right) : -kInf;
        if (bl >= br) {
            search(lo, mid);
            search(mid + 1, hi);
        } else {
            search(mid + 1, hi);
            search(lo, mid);
        }
    }

    const std::vector<std::array<double, 3>>* p_ = nullptr;
    const std::vector<double>* key_ = nullptr;
    const std::vector<int64_t>* id_ = nullptr;
    int m_ = 0;
    std::array<double, 3> a_{};
    double c_ = 0.0, best_ = -kInf;
    int best_point_ = -1;
    std::vector<int> perm_, pos_;
    std::vector<std::array<double, 3>> lo_, hi_;
    std::vector<double> best_key_;
    std::vector<char> alive_;
    std::vector<std::pair<int, int>> path_;
};

void check_order_inputs(const ArrI& ix, const ArrI& iy, const ArrI& iz, const ArrI& phase, const ArrD& key, double c)
{
    const int64_t n = key.size();
    check_size(ix, n, "ix");
    check_size(iy, n, "iy");
    check_size(iz, n, "iz");
    check_size(phase, n, "phase");
    if (!(c >= 0) || !std::isfinite(c)) throw std::invalid_argument("the proximity weight must be finite and >= 0");
    if (n >= std::numeric_limits<int32_t>::max()) throw std::invalid_argument("too many blocks for one sequence");
    const double* pk = key.data();
    for (int64_t i = 0; i < n; ++i)
        if (!std::isfinite(pk[i])) throw std::invalid_argument("keys must be finite");
}

std::vector<std::vector<std::array<int64_t, 3>>> read_patterns(const std::vector<ArrI>& offsets)
{
    if (offsets.empty()) throw std::invalid_argument("offsets must hold one (k, 3) array per domain");
    std::vector<std::vector<std::array<int64_t, 3>>> patterns;
    for (const auto& o : offsets) {
        if (o.ndim() != 2 || o.shape(1) != 3) throw std::invalid_argument("every offsets array must be (k, 3)");
        auto r = o.unchecked<2>();
        std::vector<std::array<int64_t, 3>> p;
        for (py::ssize_t k = 0; k < o.shape(0); ++k) {
            if (r(k, 2) <= 0) throw std::invalid_argument("offsets must point upwards (dz > 0)");
            p.push_back({r(k, 0), r(k, 1), r(k, 2)});
        }
        patterns.push_back(std::move(p));
    }
    return patterns;
}

// Domain of every cell (nullptr: one pattern); `holder` keeps the array alive.
const int64_t* read_domains(const py::object& domain_obj, int64_t cells, size_t n_patterns, ArrI& holder)
{
    if (domain_obj.is_none()) return nullptr;
    holder = domain_obj.cast<ArrI>();
    check_size(holder, cells, "domain");
    const int64_t* pd = holder.data();
    for (int64_t i = 0; i < cells; ++i)
        if (pd[i] < 0 || pd[i] >= static_cast<int64_t>(n_patterns)) throw std::invalid_argument("domain out of range");
    return pd;
}

// Bench order: by phase, then bench from the top (iz decreasing), and inside every phase-bench the block of
// highest key first, then repeatedly the block that maximises key - c * distance to the last one (key alone
// when c == 0). Returns the input indices in extraction order.
py::array_t<int64_t> order(const ArrI& ix, const ArrI& iy, const ArrI& iz, const ArrI& phase, const ArrD& key,
                           double c, double dx, double dy)
{
    check_order_inputs(ix, iy, iz, phase, key, c);
    const int64_t n = key.size();
    const int64_t *px = ix.data(), *py_ = iy.data(), *pz = iz.data(), *pp = phase.data();
    const double* pk = key.data();
    py::array_t<int64_t> result(n);
    {
        py::gil_scoped_release release;
        std::vector<int64_t> idx(static_cast<size_t>(n));
        std::iota(idx.begin(), idx.end(), int64_t{0});
        std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
            if (pp[a] != pp[b]) return pp[a] < pp[b];
            if (pz[a] != pz[b]) return pz[a] > pz[b];
            if (pk[a] != pk[b]) return pk[a] > pk[b];
            return a < b;
        });
        int64_t* out = result.mutable_data();
        int64_t w = 0;
        Search search;
        std::vector<std::array<double, 3>> pts;
        std::vector<double> keys;
        std::vector<int64_t> ids;
        for (int64_t a = 0; a < n;) {
            int64_t b = a + 1;
            while (b < n && pp[idx[b]] == pp[idx[a]] && pz[idx[b]] == pz[idx[a]]) ++b;
            if (c == 0.0 || b - a <= 2) {
                for (int64_t j = a; j < b; ++j) out[w++] = idx[j];
            } else {
                pts.clear();
                keys.clear();
                ids.clear();
                for (int64_t j = a; j < b; ++j) {
                    const int64_t i = idx[j];
                    pts.push_back({static_cast<double>(px[i]) * dx, static_cast<double>(py_[i]) * dy, 0.0});
                    keys.push_back(pk[i]);
                    ids.push_back(i);
                }
                search.build(pts, keys, ids, true);
                int current = 0;  // idx is sorted by key: the first one has the highest key
                for (int64_t step = a; step < b; ++step) {
                    out[w++] = ids[static_cast<size_t>(current)];
                    search.remove(current);
                    if (step + 1 < b) current = search.query(pts[static_cast<size_t>(current)], c);
                }
            }
            a = b;
        }
    }
    return result;
}

// Precedence order (the lookahead search): phase by phase; inside a phase, the next block is the AVAILABLE block
// (every predecessor mined) that maximises key - c * distance to the last one (3-D), whatever its bench; the
// first block of a phase is the available one of highest key. Availability follows the grid's precedence
// patterns over the upward closure of the blocks: empty cells are free as soon as their own predecessors are
// mined, so constraints that travel through them in a chain hold. cell: grid cell of every block; offsets: one
// (k, 3) array per domain; domain: per cell (None: one pattern); others: cells of the blocks that are not in the
// sequence (reaching one means the blocks are not a closed pit: error). Returns the input indices in order.
py::array_t<int64_t> order_free(int64_t nx, int64_t ny, int64_t nz, const ArrI& cell, const ArrI& ix, const ArrI& iy,
                                const ArrI& iz, const ArrI& phase, const ArrD& key, double c, double dx, double dy,
                                double dz, const std::vector<ArrI>& offsets, const py::object& domain_obj,
                                const ArrI& others)
{
    check_order_inputs(ix, iy, iz, phase, key, c);
    const int64_t n = key.size();
    check_size(cell, n, "cell");
    if (others.ndim() != 1) throw std::invalid_argument("others must be a 1-D array of cells");
    const int64_t cells = nx * ny * nz;
    const auto patterns = read_patterns(offsets);
    ArrI domain;
    const int64_t* pd = read_domains(domain_obj, cells, patterns.size(), domain);
    const int64_t *pc = cell.data(), *px = ix.data(), *py_ = iy.data(), *pz = iz.data(), *pp = phase.data();
    const double* pk = key.data();
    int64_t phases = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (pc[i] < 0 || pc[i] >= cells) throw std::invalid_argument("cell out of the grid");
        if (pp[i] < 1) throw std::invalid_argument("phases must be >= 1");
        phases = std::max(phases, pp[i]);
    }
    py::array_t<int64_t> result(n);
    std::string error;
    {
        py::gil_scoped_release release;
        // local index of every cell of the upward closure: the blocks first (0..n-1), then the empty cells
        std::vector<int32_t> local(static_cast<size_t>(cells), -1);
        std::vector<int64_t> region;
        region.reserve(static_cast<size_t>(n) * 2);
        for (int64_t i = 0; i < n && error.empty(); ++i) {
            if (local[static_cast<size_t>(pc[i])] >= 0) error = "two blocks in one cell";
            local[static_cast<size_t>(pc[i])] = static_cast<int32_t>(i);
            region.push_back(pc[i]);
        }
        const int64_t* po = others.data();
        for (int64_t i = 0; i < others.size() && error.empty(); ++i) {
            if (po[i] < 0 || po[i] >= cells || local[static_cast<size_t>(po[i])] >= 0)
                error = "others must be cells of the grid without a block of the sequence";
            else
                local[static_cast<size_t>(po[i])] = -2;
        }
        auto pattern_of = [&](int64_t q) -> const std::vector<std::array<int64_t, 3>>& {
            return patterns[pd ? static_cast<size_t>(pd[q]) : 0];
        };
        const int64_t layer = nx * ny;
        // calls f(predecessor cell) for every in-grid predecessor of cell q
        auto for_each_pred = [&](int64_t q, auto&& f) {
            const int64_t z = q / layer, x = q % nx, y = (q % layer) / nx;
            for (const auto& o : pattern_of(q)) {
                const int64_t xx = x + o[0], yy = y + o[1], zz = z + o[2];
                if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz >= nz) continue;
                f((zz * ny + yy) * nx + xx);
            }
        };
        std::vector<int32_t> missing;  // per cell of the closure: predecessors not mined yet
        missing.reserve(region.capacity());
        for (size_t h = 0; error.empty() && h < region.size(); ++h) {  // breadth-first, upwards
            int32_t count = 0;
            for_each_pred(region[h], [&](int64_t r) {
                ++count;
                if (local[static_cast<size_t>(r)] >= 0) return;
                if (local[static_cast<size_t>(r)] == -2) {
                    error = "the blocks are not a closed pit: a block outside the sequence is a predecessor";
                    return;
                }
                if (region.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    error = "the upward closure is too large";
                    return;
                }
                local[static_cast<size_t>(r)] = static_cast<int32_t>(region.size());
                region.push_back(r);
            });
            missing.push_back(count);
        }
        const size_t m = region.size();
        // calls f(local index) for every cell of the closure that has cell q as a predecessor (no arc list: the
        // successors are read from the patterns, a domain's offsets only for the cells of that domain)
        auto for_each_succ = [&](int64_t q, auto&& f) {
            const int64_t z = q / layer, x = q % nx, y = (q % layer) / nx;
            for (size_t d = 0; d < patterns.size(); ++d)
                for (const auto& o : patterns[d]) {
                    const int64_t xx = x - o[0], yy = y - o[1], zz = z - o[2];
                    if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0) continue;
                    const int64_t t = (zz * ny + yy) * nx + xx;
                    const int32_t lt = local[static_cast<size_t>(t)];
                    if (lt < 0 || (pd && pd[t] != static_cast<int64_t>(d))) continue;
                    f(lt);
                }
        };
        std::vector<std::vector<int64_t>> members(static_cast<size_t>(phases) + 1), pending(static_cast<size_t>(phases) + 1);
        std::vector<int32_t> slot(static_cast<size_t>(n), -1);  // position of a block inside its phase
        for (int64_t i = 0; i < n; ++i) {
            slot[static_cast<size_t>(i)] = static_cast<int32_t>(members[static_cast<size_t>(pp[i])].size());
            members[static_cast<size_t>(pp[i])].push_back(i);
        }
        int64_t* out = result.mutable_data();
        int64_t written = 0, current_phase = 1;
        Search search;
        std::vector<std::array<double, 3>> pts;
        std::vector<double> keys;
        std::vector<int64_t> ids, stack;
        std::vector<char> released(m, 0);  // each cell is released once
        // every predecessor of cell h is mined: an empty cell is mined at once, a block becomes available
        auto release_cell = [&](int64_t h) {
            stack.push_back(h);
            while (!stack.empty()) {
                const int64_t g = stack.back();
                stack.pop_back();
                if (released[static_cast<size_t>(g)]) continue;
                released[static_cast<size_t>(g)] = 1;
                if (g < n) {  // a block
                    const int64_t ph = pp[g];
                    if (ph == current_phase)
                        search.activate(slot[static_cast<size_t>(g)]);
                    else if (ph > current_phase)
                        pending[static_cast<size_t>(ph)].push_back(g);
                    else
                        error = "a block became available after its phase";
                    continue;
                }
                for_each_succ(region[static_cast<size_t>(g)], [&](int32_t t) {
                    if (--missing[static_cast<size_t>(t)] == 0) stack.push_back(t);
                });
            }
        };
        for (; error.empty() && current_phase <= phases; ++current_phase) {
            const auto& mem = members[static_cast<size_t>(current_phase)];
            pts.clear();
            keys.clear();
            ids.clear();
            for (int64_t i : mem) {
                pts.push_back({static_cast<double>(px[i]) * dx, static_cast<double>(py_[i]) * dy,
                               static_cast<double>(pz[i]) * dz});
                keys.push_back(pk[i]);
                ids.push_back(i);
            }
            search.build(pts, keys, ids, false);
            if (current_phase == 1)  // cells without predecessors are free from the start
                for (size_t h = 0; h < m && error.empty(); ++h)
                    if (missing[h] == 0) release_cell(static_cast<int64_t>(h));
            for (int64_t g : pending[static_cast<size_t>(current_phase)]) search.activate(slot[static_cast<size_t>(g)]);
            pending[static_cast<size_t>(current_phase)].clear();
            int last = -1;
            for (size_t step = 0; step < mem.size() && error.empty(); ++step) {
                const int next = last < 0 ? search.query({0.0, 0.0, 0.0}, 0.0)
                                          : search.query(pts[static_cast<size_t>(last)], c);
                if (next < 0) {
                    error = "phase " + std::to_string(current_phase) +
                            " cannot be completed: some of its blocks need blocks of later phases";
                    break;
                }
                search.remove(next);
                const int64_t b = mem[static_cast<size_t>(next)];
                out[written++] = b;
                for_each_succ(region[static_cast<size_t>(b)], [&](int32_t t) {
                    if (--missing[static_cast<size_t>(t)] == 0) release_cell(t);
                });
                last = next;
            }
        }
        if (error.empty() && written != n) error = "the sequence is incomplete";
    }
    if (!error.empty()) throw std::invalid_argument(error);
    return result;
}

// ---------------------------------------------------------------------------------------------------------------
// Continuity of the phases. A group of blocks of one phase (connected through faces, edges or corners) that is
// smaller than min_blocks and touches no block of an earlier phase is an isolated start: a rich pod mined alone,
// far from the rest of the phase. It is moved to the phase that first needs it: the lowest phase among the blocks
// that require it (found downwards through the precedence patterns, empty cells included) and its neighbours of
// later phases. Every block that requires it is then in that phase or a later one, so the bench order and the
// precedence order still respect the slopes. Rounds repeat until nothing moves. Returns (phase, groups, blocks).
py::tuple merge_small(int64_t nx, int64_t ny, int64_t nz, const ArrI& cell, const ArrI& phase_in,
                      const std::vector<ArrI>& offsets, const py::object& domain_obj, const ArrI& others,
                      int64_t min_blocks, int64_t max_rounds)
{
    const int64_t n = cell.size();
    check_size(phase_in, n, "phase");
    if (others.ndim() != 1) throw std::invalid_argument("others must be a 1-D array of cells");
    const int64_t cells = nx * ny * nz;
    const auto patterns = read_patterns(offsets);
    ArrI domain;
    const int64_t* pd = read_domains(domain_obj, cells, patterns.size(), domain);
    const int64_t *pc = cell.data(), *pin = phase_in.data(), *po = others.data();
    for (int64_t i = 0; i < n; ++i)
        if (pc[i] < 0 || pc[i] >= cells) throw std::invalid_argument("cell out of the grid");
    py::array_t<int64_t> result(n);
    int64_t moved_groups = 0, moved_blocks = 0;
    {
        py::gil_scoped_release release;
        int64_t* ph = result.mutable_data();
        std::copy(pin, pin + n, ph);
        constexpr int64_t kNone = std::numeric_limits<int64_t>::max();
        constexpr int32_t kEmpty = -1, kOther = -2;
        std::vector<int32_t> local(static_cast<size_t>(cells), kEmpty);  // block of the sequence, other block or empty
        for (int64_t i = 0; i < others.size(); ++i)
            if (po[i] >= 0 && po[i] < cells) local[static_cast<size_t>(po[i])] = kOther;
        for (int64_t i = 0; i < n; ++i) local[static_cast<size_t>(pc[i])] = static_cast<int32_t>(i);
        const int64_t layer = nx * ny;
        auto xyz = [&](int64_t q) { return std::array<int64_t, 3>{q % nx, (q % layer) / nx, q / layer}; };
        std::vector<int32_t> group(static_cast<size_t>(n), -1), members, stack;
        std::vector<int32_t> seen(static_cast<size_t>(cells), -1);  // last group that visited a cell (downward search)
        std::vector<int64_t> cell_stack;
        int32_t stamp = 0;
        for (int64_t round = 0; round < max_rounds; ++round) {
            std::fill(group.begin(), group.end(), -1);
            int64_t changes = 0;
            int32_t groups = 0;
            for (int64_t start = 0; start < n; ++start) {
                if (group[static_cast<size_t>(start)] >= 0) continue;
                const int64_t p = ph[start];
                members.clear();
                stack.assign(1, static_cast<int32_t>(start));
                group[static_cast<size_t>(start)] = groups;
                bool touches_earlier = false;
                int64_t neighbour_later = kNone;
                while (!stack.empty()) {  // the group: same phase, 26 neighbours
                    const int32_t b = stack.back();
                    stack.pop_back();
                    members.push_back(b);
                    const auto [x, y, z] = xyz(pc[b]);
                    for (int64_t dz = -1; dz <= 1; ++dz)
                        for (int64_t dy = -1; dy <= 1; ++dy)
                            for (int64_t dx = -1; dx <= 1; ++dx) {
                                const int64_t xx = x + dx, yy = y + dy, zz = z + dz;
                                if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
                                const int32_t t = local[static_cast<size_t>((zz * ny + yy) * nx + xx)];
                                if (t < 0) continue;
                                if (ph[t] < p) touches_earlier = true;
                                else if (ph[t] > p) neighbour_later = std::min(neighbour_later, ph[t]);
                                else if (group[static_cast<size_t>(t)] < 0) {
                                    group[static_cast<size_t>(t)] = groups;
                                    stack.push_back(t);
                                }
                            }
                }
                const int32_t g = groups++;
                if (touches_earlier || static_cast<int64_t>(members.size()) >= min_blocks) continue;
                // the lowest phase that requires a block of the group: downwards through the patterns; a block of
                // another group stops the search (what requires it is in its phase or a later one)
                int64_t needed = kNone;
                ++stamp;
                cell_stack.clear();
                for (int32_t b : members) {
                    seen[static_cast<size_t>(pc[b])] = stamp;
                    cell_stack.push_back(pc[b]);
                }
                while (!cell_stack.empty()) {
                    const int64_t q = cell_stack.back();
                    cell_stack.pop_back();
                    const auto [x, y, z] = xyz(q);
                    for (size_t d = 0; d < patterns.size(); ++d)
                        for (const auto& o : patterns[d]) {
                            const int64_t xx = x - o[0], yy = y - o[1], zz = z - o[2];
                            if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0) continue;
                            const int64_t t = (zz * ny + yy) * nx + xx;
                            if ((pd && pd[t] != static_cast<int64_t>(d)) || seen[static_cast<size_t>(t)] == stamp) continue;
                            seen[static_cast<size_t>(t)] = stamp;
                            const int32_t lt = local[static_cast<size_t>(t)];
                            if (lt >= 0 && group[static_cast<size_t>(lt)] != g)
                                needed = std::min(needed, ph[lt]);
                            else if (lt == kEmpty)
                                cell_stack.push_back(t);  // empty cell: what requires it requires the group
                        }
                }
                const int64_t target = std::min(needed, neighbour_later);
                if (target == kNone || target <= p) continue;
                for (int32_t b : members) ph[b] = target;
                ++changes;
                ++moved_groups;
                moved_blocks += static_cast<int64_t>(members.size());
            }
            if (changes == 0) break;
        }
    }
    return py::make_tuple(result, moved_groups, moved_blocks);
}

// ---------------------------------------------------------------------------------------------------------------
// Schedule of a sequence: period after period, blocks are taken in order as far as every capacity allows; a block
// that does not fit is split between periods (its value and capacity uses in proportion). uses is (K, n) in
// sequence order, caps (K,); track (T, n) are quantities only summed per period (e.g. tonnage, ore). Time runs
// with the binding capacity: (period - 1) + the largest fraction of a capacity used in the period. Cash of period k
// is discounted by (1 + rate) ** -(k - shift); with continuous=true every piece of a block is discounted when it is
// mined instead, by exp(-rate * t) integrated over its time. The fixed cost is paid in every
// period, the current one in proportion to its use of the binding capacity (continuously: at a constant rate).
py::dict schedule(const ArrD& value, const ArrD& uses, const ArrD& caps, const ArrD& track, double rate, double shift,
                  double fixed, bool continuous)
{
    const int64_t n = value.size();
    const int64_t K = caps.size();
    if (uses.ndim() != 2 || uses.shape(0) != K || uses.shape(1) != n)
        throw std::invalid_argument("uses must be (capacities, blocks)");
    if (track.ndim() != 2 || track.shape(1) != n) throw std::invalid_argument("track must be (quantities, blocks)");
    const int64_t T = track.shape(0);
    if (K < 1) throw std::invalid_argument("at least one capacity is needed");
    if (!std::isfinite(rate) || rate <= -1.0) throw std::invalid_argument("rate must be finite and > -1");
    const double* pv = value.data();
    const double* pw = uses.data();
    const double* pc = caps.data();
    const double* pt = track.data();
    for (int64_t c = 0; c < K; ++c)
        if (!(pc[c] > 0) || !std::isfinite(pc[c])) throw std::invalid_argument("capacities must be finite and > 0");
    for (int64_t j = 0; j < K * n; ++j)
        if (!(pw[j] >= 0) || !std::isfinite(pw[j])) throw std::invalid_argument("capacity uses must be finite and >= 0");
    py::array_t<int64_t> first_period(n);
    py::array_t<double> discounted(n), npv_stop(n), time_end(n);
    std::vector<double> period_value, period_factor, period_discounted;
    std::vector<std::vector<double>> period_use(static_cast<size_t>(K)), period_track(static_cast<size_t>(T));
    // continuous discounting: the mean of exp(-rate * t) over [a, b] (exact for a constant cash rate)
    auto mean_factor = [rate](double a, double b) {
        if (rate == 0.0) return 1.0;
        const double h = rate * (b - a);
        if (h < 1e-300) return std::exp(-rate * a);
        return std::exp(-rate * a) * (-std::expm1(-h)) / h;  // exact also for very short pieces
    };
    auto fixed_until = [rate, fixed](double t) {  // discounted fixed cost paid at a constant rate from 0 to t
        return rate == 0.0 ? fixed * t : fixed * (1.0 - std::exp(-rate * t)) / rate;
    };
    {
        py::gil_scoped_release release;
        int64_t* fp = first_period.mutable_data();
        double *dv = discounted.mutable_data(), *ns = npv_stop.mutable_data(), *te = time_end.mutable_data();
        constexpr size_t max_periods = 1000000;
        std::vector<double> used(static_cast<size_t>(K), 0.0);
        auto open = [&]() {
            if (period_value.size() >= max_periods)
                throw std::invalid_argument("the schedule needs more than 1,000,000 periods: check that the "
                                            "capacities and the tonnage are in the same units");
            const double k = static_cast<double>(period_value.size() + 1);
            period_value.push_back(0.0);
            period_discounted.push_back(0.0);
            period_factor.push_back(continuous ? 0.0 : std::pow(1.0 + rate, -(k - shift)));
            for (auto& u : period_use) u.push_back(0.0);
            for (auto& t : period_track) t.push_back(0.0);
            std::fill(used.begin(), used.end(), 0.0);
        };
        open();
        double cum = 0.0, closed_factors = 0.0;  // discounted cash; sum of the factors of closed periods
        auto clock = [&]() {  // time elapsed: closed periods plus the largest fraction of a capacity used
            double share = 0.0;
            for (int64_t c = 0; c < K; ++c) share = std::max(share, used[static_cast<size_t>(c)] / pc[c]);
            return static_cast<double>(period_value.size() - 1) + std::min(share, 1.0);
        };
        for (int64_t i = 0; i < n; ++i) {
            double rest = 1.0, disc = 0.0;
            fp[i] = 0;
            while (rest > 0.0) {
                double f = rest;
                for (int64_t c = 0; c < K; ++c) {
                    const double w = pw[c * n + i];
                    if (w > 0.0) f = std::min(f, (pc[c] - used[static_cast<size_t>(c)]) / w);
                }
                bool fresh = true;  // nothing mined yet in this period
                for (int64_t c = 0; c < K && fresh; ++c) fresh = used[static_cast<size_t>(c)] == 0.0;
                if (fresh && !(f > 0.0))
                    throw std::invalid_argument("a block uses more than 1e300 times a capacity: check the units");
                if (f <= 1e-12 && !fresh) {  // a capacity is full: next period
                    closed_factors += period_factor.back();
                    open();
                    continue;
                }
                if (fp[i] == 0) fp[i] = static_cast<int64_t>(period_value.size());
                const double t0 = clock();
                for (int64_t c = 0; c < K; ++c) {
                    const double take = f * pw[c * n + i];
                    used[static_cast<size_t>(c)] += take;
                    period_use[static_cast<size_t>(c)].back() += take;
                }
                for (int64_t t = 0; t < T; ++t) period_track[static_cast<size_t>(t)].back() += f * pt[t * n + i];
                period_value.back() += f * pv[i];
                const double piece = f * pv[i] * (continuous ? mean_factor(t0, clock()) : period_factor.back());
                period_discounted.back() += piece;
                disc += piece;
                rest = rest - f <= 1e-12 ? 0.0 : rest - f;
            }
            dv[i] = disc;
            cum += disc;
            double share = 0.0;
            for (int64_t c = 0; c < K; ++c) share = std::max(share, used[static_cast<size_t>(c)] / pc[c]);
            share = std::min(share, 1.0);
            te[i] = static_cast<double>(period_value.size() - 1) + share;
            ns[i] = cum - (continuous ? fixed_until(te[i]) : fixed * (closed_factors + share * period_factor.back()));
        }
    }
    py::dict out;
    out["first_period"] = first_period;
    out["discounted"] = discounted;
    out["npv_stop"] = npv_stop;
    out["time_end"] = time_end;
    out["period_value"] = py::array_t<double>(static_cast<py::ssize_t>(period_value.size()), period_value.data());
    out["period_factor"] = py::array_t<double>(static_cast<py::ssize_t>(period_factor.size()), period_factor.data());
    out["period_discounted"] =
        py::array_t<double>(static_cast<py::ssize_t>(period_discounted.size()), period_discounted.data());
    auto rows = [](const std::vector<std::vector<double>>& v) {
        py::list out;
        for (const auto& u : v) out.append(py::array_t<double>(static_cast<py::ssize_t>(u.size()), u.data()));
        return out;
    };
    out["period_use"] = rows(period_use);
    out["period_track"] = rows(period_track);
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Precedence check of a sequence. rank: per cell, the position of its block in the sequence (>= 0), -1 for an
// empty cell (no block) and -2 for a block that is not in the sequence. Empty cells are free once their own
// predecessors are mined, so constraints that travel through them in a chain are checked too. Returns
// (violations, first violating cell or -1).
py::tuple check(int64_t nx, int64_t ny, int64_t nz, const ArrI& rank, const std::vector<ArrI>& offsets,
                const py::object& domain_obj, int threads)
{
    const int64_t n = nx * ny * nz;
    check_size(rank, n, "rank");
    const auto patterns = read_patterns(offsets);
    ArrI domain;
    const int64_t* pd = read_domains(domain_obj, n, patterns.size(), domain);
    const int64_t* pr = rank.data();
    constexpr int64_t kNever = std::numeric_limits<int64_t>::max();
    std::vector<int64_t> eff(static_cast<size_t>(n), -1);
    int64_t violations = 0, first = -1;
    {
        py::gil_scoped_release release;
        const int64_t layer = nx * ny;
        std::vector<int64_t> bad(static_cast<size_t>(layer), 0);
        for (int64_t z = nz - 1; z >= 0; --z) {
            parallel_for(layer, threads, [&](int64_t a, int64_t b) {
                for (int64_t q = a; q < b; ++q) {
                    const int64_t cell = z * layer + q, x = q % nx, y = q / nx;
                    const auto& pattern = patterns[pd ? static_cast<size_t>(pd[cell]) : 0];
                    int64_t need = -1;  // latest position among the predecessors
                    for (const auto& o : pattern) {
                        const int64_t xx = x + o[0], yy = y + o[1], zz = z + o[2];
                        if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz >= nz) continue;
                        need = std::max(need, eff[static_cast<size_t>((zz * ny + yy) * nx + xx)]);
                    }
                    const int64_t r = pr[cell];
                    bad[static_cast<size_t>(q)] = 0;
                    if (r >= 0) {
                        bad[static_cast<size_t>(q)] = need >= r;
                        eff[static_cast<size_t>(cell)] = r;
                    } else if (r == -1) {
                        eff[static_cast<size_t>(cell)] = need;
                    } else {
                        eff[static_cast<size_t>(cell)] = kNever;
                    }
                }
            });
            for (int64_t q = 0; q < layer; ++q)
                if (bad[static_cast<size_t>(q)]) {
                    if (first < 0) first = z * layer + q;
                    ++violations;
                }
        }
    }
    return py::make_tuple(violations, first);
}

}  // namespace

void register_oes(py::module_& m)
{
    m.def("_oes_lookahead", &lookahead, py::arg("ix"), py::arg("iy"), py::arg("iz"), py::arg("value"), py::arg("dx"),
          py::arg("dy"), py::arg("radii"), py::arg("weights"), py::arg("threads") = 0,
          "Lookahead value of every block: its value plus weights[k-1] times the value of the blocks k benches below "
          "within radii[k-1] (centre to centre, boundary included). Releases the GIL.");
    m.def("_oes_order", &order, py::arg("ix"), py::arg("iy"), py::arg("iz"), py::arg("phase"), py::arg("key"),
          py::arg("c"), py::arg("dx"), py::arg("dy"),
          "Bench order: phase, bench from the top, then the proximity walk (key - c * distance). Releases the GIL.");
    m.def("_oes_order_free", &order_free, py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("cell"), py::arg("ix"),
          py::arg("iy"), py::arg("iz"), py::arg("phase"), py::arg("key"), py::arg("c"), py::arg("dx"), py::arg("dy"),
          py::arg("dz"), py::arg("offsets"), py::arg("domain"), py::arg("others"),
          "Precedence order: phase by phase, the available block of highest key - c * distance. Releases the GIL.");
    m.def("_oes_merge_small", &merge_small, py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("cell"),
          py::arg("phase"), py::arg("offsets"), py::arg("domain"), py::arg("others"), py::arg("min_blocks"),
          py::arg("max_rounds") = 50,
          "Isolated groups of a phase smaller than min_blocks move to the phase that first needs them: "
          "(phase, groups, blocks). Releases the GIL.");
    m.def("_oes_schedule", &schedule, py::arg("value"), py::arg("uses"), py::arg("caps"), py::arg("track"),
          py::arg("rate"), py::arg("shift"), py::arg("fixed"), py::arg("continuous") = false,
          "Period-by-period fill of a sequence under capacities. Releases the GIL.");
    m.def("_oes_check", &check, py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("rank"), py::arg("offsets"),
          py::arg("domain") = py::none(), py::arg("threads") = 0,
          "(violations, first violating cell) of a sequence against precedence patterns. Releases the GIL.");
}
