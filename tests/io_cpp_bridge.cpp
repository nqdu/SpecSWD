#include "shared/io.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

specswd::io::DispersionTable example_table()
{
    specswd::io::DispersionTable table;
    table.frequency_hz = {0.1,0.2};
    table.azimuth_deg = {10.,20.};
    table.phase_velocity.indptr = {0,2,2,3,6};
    table.phase_velocity.mode_index = {0,2,1,0,1,4};
    table.phase_velocity.values = {
        {3.0,0.1},{3.2,0.2},{3.5,0.3},
        {4.0,0.4},{4.1,0.5},{4.4,0.6}
    };
    table.phase_velocity.component_names = {"phase"};
    table.group_velocity.indptr = {0,1,3,3,4};
    table.group_velocity.mode_index = {0,0,3,1};
    table.group_velocity.values = {
        {2.0,0.1},{0.2,0.01},
        {2.1,0.2},{0.3,0.02},
        {2.3,0.3},{0.4,0.03},
        {2.5,0.4},{0.5,0.04}
    };
    table.group_velocity.component_names = {"x","y"};
    return table;
}

specswd::io::SolverOutput example_output()
{
    specswd::io::SolverOutput output;
    output.dispersion.frequency_hz = {0.1,0.2};
    output.dispersion.azimuth_deg = {0.};
    output.dispersion.phase_velocity.indptr = {0,2,3};
    output.dispersion.phase_velocity.mode_index = {0,2,1};
    output.dispersion.phase_velocity.values = {
        {3.0,0.1},{3.2,0.2},{3.5,0.3}
    };
    output.dispersion.phase_velocity.component_names = {"phase"};
    output.dispersion.group_velocity.indptr = {0,2,3};
    output.dispersion.group_velocity.mode_index = {0,2,1};
    output.dispersion.group_velocity.values = {
        {2.8,0.1},{3.0,0.2},{3.2,0.3}
    };
    output.dispersion.group_velocity.component_names = {"radial"};
    output.wave_type = 0;
    output.kernel_type = 0;
    output.kernel_observable = "phase_velocity";
    output.model_depth = {0.,10.};
    output.sem_depth_indptr = {0,2,5};
    output.sem_depth_values = {0.,20.,0.,10.,30.};
    output.eigenfunctions.indptr = {0,2,4,7};
    output.eigenfunctions.component_names = {"transverse"};
    output.eigenfunctions.units = "normalized";
    output.eigenfunctions.values = {
        {1.,0.},{0.5,0.1},{1.,0.},{0.4,0.2},
        {1.,0.},{0.3,0.1},{0.1,0.}
    };
    output.elastic_kernels.parameter_names = {"vsh"};
    output.elastic_kernels.velocity = {1.,2.,3.,4.,5.,6.};
    return output;
}

template<class T>
void require_equal(const T &actual, const T &expected, const char *name)
{
    if(actual!=expected) throw std::runtime_error(std::string(name)+" differs");
}

void check(const specswd::io::DispersionTable &actual)
{
    const auto expected = example_table();
    require_equal(actual.frequency_hz,expected.frequency_hz,"frequency_hz");
    require_equal(actual.azimuth_deg,expected.azimuth_deg,"azimuth_deg");
    require_equal(
        actual.phase_velocity.indptr,expected.phase_velocity.indptr,
        "phase indptr"
    );
    require_equal(
        actual.phase_velocity.mode_index,expected.phase_velocity.mode_index,
        "phase mode_index"
    );
    require_equal(
        actual.phase_velocity.values,expected.phase_velocity.values,
        "phase values"
    );
    require_equal(
        actual.group_velocity.indptr,expected.group_velocity.indptr,
        "group indptr"
    );
    require_equal(
        actual.group_velocity.mode_index,expected.group_velocity.mode_index,
        "group mode_index"
    );
    require_equal(
        actual.group_velocity.values,expected.group_velocity.values,
        "group values"
    );
}

} // namespace

int main(int argc, char **argv)
{
    try {
        if(argc!=3) throw std::invalid_argument(
            "usage: io_cpp_bridge write|write-full|check file");
        const std::string action(argv[1]);
        if(action=="write") specswd::io::write_hdf5(argv[2],example_table());
        else if(action=="write-full") {
            specswd::io::write_hdf5(argv[2],example_output());
        }
        else if(action=="check") check(specswd::io::read_hdf5(argv[2]));
        else throw std::invalid_argument("unknown action");
        return 0;
    }
    catch(const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
