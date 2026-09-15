#include "vti/vti.hpp"
#include "shared/GQTable.hpp"
#include "shared/io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using specswd::Complex;
using specswd::Real;

std::vector<Real> logarithmic_frequencies(Real first, Real last, int count)
{
    if(!std::isfinite(first) || !std::isfinite(last) || first<=0. || last<=0.) {
        throw std::invalid_argument("frequencies must be finite and positive");
    }
    if(count<=0) throw std::invalid_argument("nt must be positive");
    if(first>last) std::swap(first,last);
    std::vector<Real> values(static_cast<std::size_t>(count));
    const Real start = std::log10(first);
    const Real stop = std::log10(last);
    const Real denominator = count==1 ? 1. : static_cast<Real>(count-1);
    for(int i=0;i<count;++i) {
        values[static_cast<std::size_t>(i)] =
            std::pow(10.,start+(stop-start)*i/denominator);
    }
    return values;
}

void append_physical_depth(
    specswd::io::SolverOutput &output,
    const specswd::Mesh &mesh
)
{
    for(const Real depth:mesh.znodes) {
        output.sem_depth_values.push_back(depth*mesh.SCALE_LENGTH);
    }
    output.sem_depth_indptr.push_back(output.sem_depth_values.size());
}

void append_eigenfunction(
    specswd::io::RaggedComplexField &field,
    const std::vector<Complex> &values
)
{
    field.values.insert(field.values.end(),values.begin(),values.end());
    field.indptr.push_back(field.npoint());
}

void append_projected(
    const specswd::Mesh &mesh,
    const std::vector<Real> &sem,
    int nparameter,
    std::vector<Real> &destination
)
{
    const std::size_t npoint = mesh.ibool_el.size();
    if(sem.size()!=static_cast<std::size_t>(nparameter)*npoint) {
        throw std::runtime_error("Love-wave kernel has an inconsistent SEM shape");
    }
    const std::size_t offset = destination.size();
    destination.resize(offset+static_cast<std::size_t>(nparameter)*mesh.nz_tomo);
    for(int parameter=0;parameter<nparameter;++parameter) {
        mesh.project_kl(sem.data()+static_cast<std::size_t>(parameter)*npoint,
                        destination.data()+offset+
                            static_cast<std::size_t>(parameter)*mesh.nz_tomo);
    }
}

} // namespace

int main(int argc, char **argv)
{
    try {
        if(argc!=5 && argc!=6) {
            throw std::invalid_argument(
                "usage: surflove modelfile f1 f2 nt [KERNEL_TYPE=1]");
        }
        GQTable::initialize();

        auto mesh = std::make_unique<specswd::Mesh>();
        mesh->read_model(argv[1]);
        mesh->create_model_attributes();
        if(mesh->SWD_TYPE!=0) {
            throw std::invalid_argument("surflove requires a Love-wave model");
        }
        mesh->print_model();

        const auto frequencies = logarithmic_frequencies(
            std::stod(argv[2]),std::stod(argv[3]),std::stoi(argv[4])
        );
        const int kernel_type = argc==6 ? std::stoi(argv[5]):1;
        if(kernel_type!=0 && kernel_type!=1) {
            throw std::invalid_argument("KERNEL_TYPE must be 0 (phase) or 1 (group)");
        }

        auto solver = std::make_unique<specswd::SolverLove>();
        solver->build(mesh.get());

        specswd::io::SolverOutput output;
        output.wave_type = mesh->SWD_TYPE;
        output.attenuation = mesh->HAS_ATT;
        output.kernel_type = kernel_type;
        output.kernel_observable = kernel_type==0 ?
            "phase_velocity":"radial_group_velocity";
        output.dispersion.frequency_hz = frequencies;
        output.dispersion.azimuth_deg = {0.};
        output.dispersion.phase_velocity.component_names = {"phase"};
        output.dispersion.phase_velocity.indptr = {0};
        output.dispersion.group_velocity.component_names = {"radial"};
        output.dispersion.group_velocity.indptr = {0};
        output.sem_depth_indptr = {0};
        output.eigenfunctions.component_names = {"transverse"};
        output.eigenfunctions.units = "normalized";
        output.eigenfunctions.indptr = {0};
        output.elastic_kernels.parameter_names = {"vsh","vsv","rho"};
        if(mesh->HAS_ATT) {
            output.elastic_kernels.parameter_names.insert(
                output.elastic_kernels.parameter_names.end(),{"Qn","Ql"}
            );
        }
        output.model_depth.reserve(mesh->depth_tomo.size());
        for(const Real depth:mesh->depth_tomo) {
            output.model_depth.push_back(depth*mesh->SCALE_LENGTH);
        }

        for(const Real frequency:frequencies) {
            mesh->create_database(frequency,0.);
            append_physical_depth(output,*mesh);
            solver->prepare_matrices();
            solver->compute_egn(true);
            solver->compute_group_vel();

            const int mode_count = static_cast<int>(solver->c_phase.size());
            for(int mode=0;mode<mode_count;++mode) {
                Real real{},imag{};
                solver->get_phase_vel(mode,real,imag);
                output.dispersion.phase_velocity.mode_index.push_back(mode);
                output.dispersion.phase_velocity.values.emplace_back(real,imag);
                solver->get_group_vel(mode,real,imag);
                output.dispersion.group_velocity.mode_index.push_back(mode);
                output.dispersion.group_velocity.values.emplace_back(real,imag);

                std::vector<Complex> displacement(mesh->ibool_el.size());
                solver->egn2displ(mode,displacement.data());
                append_eigenfunction(output.eigenfunctions,displacement);

                std::vector<Real> velocity_kernel,quality_kernel;
                solver->compute_kernels(mode,kernel_type,velocity_kernel,
                                        quality_kernel);
                append_projected(*mesh,velocity_kernel,solver->nkers,
                                 output.elastic_kernels.velocity);
                if(mesh->HAS_ATT) {
                    append_projected(
                        *mesh,quality_kernel,solver->nkers,
                        output.elastic_kernels.propagation_q_inverse
                    );
                }
            }
            output.dispersion.phase_velocity.indptr.push_back(
                output.dispersion.phase_velocity.mode_index.size());
            output.dispersion.group_velocity.indptr.push_back(
                output.dispersion.group_velocity.mode_index.size());
        }

        std::filesystem::create_directories("out");
        specswd::io::write_hdf5("out/specswd.h5",output);
        std::cout << "Wrote out/specswd.h5\n";
        return 0;
    }
    catch(const std::exception &error) {
        std::cerr << "surflove: " << error.what() << '\n';
        return 1;
    }
}
