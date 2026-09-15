#ifndef SPECSWD_SHARED_IO_HPP_
#define SPECSWD_SHARED_IO_HPP_

#include "numerical.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace specswd::io
{

/**
 * A complex-valued CSR table whose rows are frequency/azimuth solve points.
 * Values are row-major with shape (nentry,ncomponent).
 */
struct ComplexCSR
{
    std::vector<std::uint64_t> indptr;
    std::vector<std::int32_t> mode_index;
    std::vector<Complex> values;
    std::vector<std::string> component_names;

    static ComplexCSR from_rows(
        const std::vector<std::vector<Complex>> &rows,
        std::vector<std::string> component_names,
        const std::vector<std::vector<std::int32_t>> &mode_rows = {}
    );

    std::size_t ncomponent() const noexcept;
    std::size_t nentry() const noexcept;
    std::pair<std::size_t,std::size_t> row_range(std::size_t row) const;
    void validate(std::size_t nrow, const char *name) const;
};

/**
 * Ragged dispersion values on a frequency/azimuth grid.
 *
 * Phase and group data have independent CSR structures so callers may save
 * different mode selections for the two observables. Solve-point rows use
 * azimuth as the inner dimension:
 *     row = frequency_index*n_azimuth + azimuth_index.
 */
struct DispersionTable
{
    std::vector<Real> frequency_hz;
    std::vector<Real> azimuth_deg;
    ComplexCSR phase_velocity;
    ComplexCSR group_velocity;
    std::string velocity_units = "km/s";

    std::size_t nsolve() const noexcept;
    bool has_group_velocity() const noexcept;
    void validate() const;
};

/** A variable-length complex field with one row per dispersion entry. */
struct RaggedComplexField
{
    std::vector<std::uint64_t> indptr;
    std::vector<Complex> values;
    std::vector<std::string> component_names;
    std::string units;

    std::size_t ncomponent() const noexcept;
    std::size_t npoint() const noexcept;
    void validate(std::size_t nrow, const char *name) const;
};

/** Projected kernels stored as (dispersion entry, parameter, model depth). */
struct KernelTable
{
    std::vector<std::string> parameter_names;
    std::vector<Real> velocity;
    std::vector<Real> propagation_q_inverse;

    bool empty() const noexcept;
    void validate(
        std::size_t nentry,
        std::size_t ndepth,
        bool attenuation,
        const char *name
    ) const;
};

/** Complete output produced by the command-line solvers. */
struct SolverOutput
{
    DispersionTable dispersion;
    int wave_type = -1;
    bool attenuation = false;
    int kernel_type = 0;
    std::string kernel_observable = "phase";

    std::vector<Real> model_depth;
    std::vector<std::uint64_t> sem_depth_indptr;
    std::vector<Real> sem_depth_values;
    RaggedComplexField eigenfunctions;
    KernelTable elastic_kernels;
    KernelTable acoustic_kernels;

    void validate() const;
};

void write_hdf5(const std::string &path, const DispersionTable &table);
void write_hdf5(const std::string &path, const SolverOutput &output);
DispersionTable read_hdf5(const std::string &path);

} // namespace specswd::io

#endif
