#include "aniso/aniso.hpp"
#include "shared/GQTable.hpp"
#include "shared/io.hpp"

#include <algorithm>
#include <cmath>
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

void append_component_major_field(
    specswd::io::RaggedComplexField &field,
    const std::vector<Complex> &values,
    std::size_t npoint
)
{
    const std::size_t ncomponent = field.ncomponent();
    if(values.size()!=npoint*ncomponent) {
        throw std::runtime_error("eigenfunction has an inconsistent shape");
    }
    for(std::size_t point=0;point<npoint;++point) {
        for(std::size_t component=0;component<ncomponent;++component) {
            field.values.push_back(values[component*npoint+point]);
        }
    }
    field.indptr.push_back(field.npoint());
}

void append_projected(
    const specswd::Mesh &mesh,
    const std::vector<Real> &sem,
    int nparameter,
    std::vector<Real> &destination
)
{
    const std::size_t npoint = mesh.ibool.size();
    if(sem.size()!=static_cast<std::size_t>(nparameter)*npoint) {
        throw std::runtime_error("anisotropic kernel has an inconsistent SEM shape");
    }
    const std::size_t offset = destination.size();
    destination.resize(offset+static_cast<std::size_t>(nparameter)*mesh.nz_tomo);
    for(int parameter=0;parameter<nparameter;++parameter) {
        mesh.project_kl(sem.data()+static_cast<std::size_t>(parameter)*npoint,
                        destination.data()+offset+
                            static_cast<std::size_t>(parameter)*mesh.nz_tomo);
    }
}

std::vector<std::string> elastic_parameter_names(bool attenuation)
{
    std::vector<std::string> names;
    names.reserve(24);
    for(int i=1;i<=6;++i) {
        for(int j=i;j<=6;++j) names.push_back("c"+std::to_string(i)+std::to_string(j));
    }
    names.push_back("rho");
    if(attenuation) {
        names.push_back("Qk");
        names.push_back("Qm");
    }
    return names;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        if(argc!=6 && argc!=7) {
            throw std::invalid_argument(
                "usage: surfani modelfile f1 f2 nt phi [KERNEL_TYPE=1]");
        }
        GQTable::initialize();

        auto mesh = std::make_unique<specswd::Mesh>();
        mesh->read_model(argv[1]);
        mesh->create_model_attributes();
        if(mesh->SWD_TYPE!=2) {
            throw std::invalid_argument("surfani requires a fully anisotropic model");
        }
        mesh->print_model();

        const auto frequencies = logarithmic_frequencies(
            std::stod(argv[2]),std::stod(argv[3]),std::stoi(argv[4])
        );
        const Real azimuth = std::stod(argv[5]);
        if(!std::isfinite(azimuth)) {
            throw std::invalid_argument("phi must be finite");
        }
        const int kernel_type = argc==7 ? std::stoi(argv[6]):1;
        if(kernel_type!=0 && kernel_type!=1) {
            throw std::invalid_argument("KERNEL_TYPE must be 0 (phase) or 1 (group)");
        }

        auto solver = std::make_unique<specswd::SolverAniso>();
        solver->build(mesh.get());

        specswd::io::SolverOutput output;
        output.wave_type = mesh->SWD_TYPE;
        output.attenuation = mesh->HAS_ATT;
        output.kernel_type = kernel_type;
        output.kernel_observable = kernel_type==0 ?
            "phase_velocity":"radial_group_velocity";
        output.dispersion.frequency_hz = frequencies;
        output.dispersion.azimuth_deg = {azimuth};
        output.dispersion.phase_velocity.component_names = {"phase"};
        output.dispersion.phase_velocity.indptr = {0};
        output.dispersion.group_velocity.component_names = {"x","y"};
        output.dispersion.group_velocity.indptr = {0};
        output.sem_depth_indptr = {0};
        output.eigenfunctions.component_names = {"x","y","z"};
        output.eigenfunctions.units = "normalized";
        output.eigenfunctions.indptr = {0};
        output.elastic_kernels.parameter_names =
            elastic_parameter_names(mesh->HAS_ATT);
        output.acoustic_kernels.parameter_names = {"kappa_ac","rho_ac"};
        if(mesh->HAS_ATT) {
            output.acoustic_kernels.parameter_names.push_back("Qk_ac");
        }
        if(output.elastic_kernels.parameter_names.size()!=
           static_cast<std::size_t>(solver->nkers_el)) {
            throw std::runtime_error(
                "unsupported anisotropic attenuation parameterization");
        }
        output.model_depth.reserve(mesh->depth_tomo.size());
        for(const Real depth:mesh->depth_tomo) {
            output.model_depth.push_back(depth*mesh->SCALE_LENGTH);
        }

        for(const Real frequency:frequencies) {
            mesh->create_database(frequency,azimuth);
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

                Real ux_real{},ux_imag{},uy_real{},uy_imag{};
                solver->get_group_vel(mode,ux_real,ux_imag,uy_real,uy_imag);
                output.dispersion.group_velocity.mode_index.push_back(mode);
                output.dispersion.group_velocity.values.emplace_back(
                    ux_real,ux_imag
                );
                output.dispersion.group_velocity.values.emplace_back(
                    uy_real,uy_imag
                );

                const std::size_t npoint = mesh->ibool.size();
                std::vector<Complex> displacement(3*npoint);
                solver->egn2displ(mode,displacement.data());
                append_component_major_field(output.eigenfunctions,displacement,
                                             npoint);

                std::vector<Real> elastic_velocity,elastic_quality;
                std::vector<Real> acoustic_velocity,acoustic_quality;
                solver->compute_kernels(
                    mode,kernel_type,elastic_velocity,elastic_quality,
                    acoustic_velocity,acoustic_quality,-1
                );
                append_projected(*mesh,elastic_velocity,solver->nkers_el,
                                 output.elastic_kernels.velocity);
                append_projected(*mesh,acoustic_velocity,solver->nkers_ac,
                                 output.acoustic_kernels.velocity);
                if(mesh->HAS_ATT) {
                    append_projected(
                        *mesh,elastic_quality,solver->nkers_el,
                        output.elastic_kernels.propagation_q_inverse
                    );
                    append_projected(
                        *mesh,acoustic_quality,solver->nkers_ac,
                        output.acoustic_kernels.propagation_q_inverse
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
        std::cerr << "surfani: " << error.what() << '\n';
        return 1;
    }
}
