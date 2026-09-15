#include "gtest/gtest.h"
#include "shared/io.hpp"

#include <cstdio>

using specswd::Complex;
using specswd::io::ComplexCSR;
using specswd::io::DispersionTable;

namespace
{

DispersionTable example_table()
{
    DispersionTable table;
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

} // namespace

TEST(DispersionIO, CSRRowsAndHDF5RoundTrip)
{
    const auto expected = example_table();
    expected.validate();
    EXPECT_EQ(expected.nsolve(),4u);
    EXPECT_EQ(
        expected.phase_velocity.row_range(1),
        std::make_pair(std::size_t{2},std::size_t{2})
    );
    EXPECT_EQ(
        expected.phase_velocity.row_range(3),
        std::make_pair(std::size_t{3},std::size_t{6})
    );

    const std::string path = ::testing::TempDir()+"specswd_io_roundtrip.h5";
    specswd::io::write_hdf5(path,expected);
    const auto actual = specswd::io::read_hdf5(path);
    std::remove(path.c_str());

    EXPECT_EQ(actual.frequency_hz,expected.frequency_hz);
    EXPECT_EQ(actual.azimuth_deg,expected.azimuth_deg);
    EXPECT_EQ(actual.phase_velocity.indptr,expected.phase_velocity.indptr);
    EXPECT_EQ(actual.phase_velocity.mode_index,expected.phase_velocity.mode_index);
    EXPECT_EQ(actual.phase_velocity.values,expected.phase_velocity.values);
    EXPECT_EQ(actual.group_velocity.indptr,expected.group_velocity.indptr);
    EXPECT_EQ(actual.group_velocity.mode_index,expected.group_velocity.mode_index);
    EXPECT_EQ(actual.group_velocity.values,expected.group_velocity.values);
    EXPECT_EQ(
        actual.group_velocity.component_names,
        expected.group_velocity.component_names
    );
}

TEST(DispersionIO, RejectsInvalidCSR)
{
    auto table = example_table();
    table.phase_velocity.indptr.back() = 20;
    EXPECT_THROW(table.validate(),std::invalid_argument);
}

TEST(DispersionIO, PacksVariableLengthRows)
{
    const auto csr = ComplexCSR::from_rows(
        {{{3.0,0.1},{3.2,0.2}},{},{{4.0,0.3}}},
        {"phase"},
        {{0,2},{},{1}}
    );
    EXPECT_EQ(csr.indptr,(std::vector<std::uint64_t>{0,2,2,3}));
    EXPECT_EQ(csr.mode_index,(std::vector<std::int32_t>{0,2,1}));
    EXPECT_EQ(csr.values.size(),3u);
}
