#include "shared/io.hpp"

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace specswd::io
{
namespace
{

struct FileComplex
{
    double r;
    double i;
};

class H5Handle
{
public:
    H5Handle() = default;
    H5Handle(hid_t id, herr_t (*close)(hid_t)):id_(id),close_(close) {}
    H5Handle(const H5Handle &) = delete;
    H5Handle &operator=(const H5Handle &) = delete;
    H5Handle(H5Handle &&other) noexcept:
        id_(std::exchange(other.id_,-1)),close_(other.close_) {}
    H5Handle &operator=(H5Handle &&other) noexcept
    {
        if(this!=&other) {
            reset();
            id_ = std::exchange(other.id_,-1);
            close_ = other.close_;
        }
        return *this;
    }
    ~H5Handle() { reset(); }
    hid_t get() const noexcept { return id_; }

private:
    void reset() noexcept
    {
        if(id_>=0 && close_) close_(id_);
        id_ = -1;
    }
    hid_t id_ = -1;
    herr_t (*close_)(hid_t) = nullptr;
};

hid_t require_id(hid_t id, const std::string &operation)
{
    if(id<0) throw std::runtime_error(operation);
    return id;
}

void require_status(herr_t status, const std::string &operation)
{
    if(status<0) throw std::runtime_error(operation);
}

H5Handle scalar_space()
{
    return {require_id(H5Screate(H5S_SCALAR),"cannot create scalar dataspace"),
            H5Sclose};
}

H5Handle string_type()
{
    H5Handle type(require_id(H5Tcopy(H5T_C_S1),"cannot copy string type"),
                  H5Tclose);
    require_status(H5Tset_size(type.get(),H5T_VARIABLE),
                   "cannot configure variable-length string type");
    require_status(H5Tset_cset(type.get(),H5T_CSET_UTF8),
                   "cannot configure UTF-8 string type");
    return type;
}

H5Handle complex_type(hid_t scalar_type)
{
    H5Handle type(require_id(H5Tcreate(H5T_COMPOUND,sizeof(FileComplex)),
                             "cannot create complex compound type"),H5Tclose);
    require_status(H5Tinsert(type.get(),"r",HOFFSET(FileComplex,r),scalar_type),
                   "cannot add real member to complex type");
    require_status(H5Tinsert(type.get(),"i",HOFFSET(FileComplex,i),scalar_type),
                   "cannot add imaginary member to complex type");
    return type;
}

void write_string_attribute(
    hid_t object,
    const char *name,
    const std::string &value
)
{
    auto type = string_type();
    auto space = scalar_space();
    H5Handle attribute(
        require_id(H5Acreate2(object,name,type.get(),space.get(),H5P_DEFAULT,
                              H5P_DEFAULT),
                   std::string("cannot create attribute ")+name),
        H5Aclose
    );
    const char *text = value.c_str();
    require_status(H5Awrite(attribute.get(),type.get(),&text),
                   std::string("cannot write attribute ")+name);
}

std::string read_string_attribute(hid_t object, const char *name)
{
    H5Handle attribute(
        require_id(H5Aopen(object,name,H5P_DEFAULT),
                   std::string("cannot open attribute ")+name),
        H5Aclose
    );
    auto type = string_type();
    char *raw = nullptr;
    require_status(H5Aread(attribute.get(),type.get(),&raw),
                   std::string("cannot read attribute ")+name);
    std::string value(raw ? raw : "");
    if(raw) H5free_memory(raw);
    return value;
}

void write_integer_attribute(hid_t object, const char *name, int value)
{
    auto space = scalar_space();
    H5Handle attribute(
        require_id(H5Acreate2(object,name,H5T_STD_I32LE,space.get(),H5P_DEFAULT,
                              H5P_DEFAULT),
                   std::string("cannot create attribute ")+name),
        H5Aclose
    );
    require_status(H5Awrite(attribute.get(),H5T_NATIVE_INT,&value),
                   std::string("cannot write attribute ")+name);
}

int read_integer_attribute(hid_t object, const char *name)
{
    H5Handle attribute(
        require_id(H5Aopen(object,name,H5P_DEFAULT),
                   std::string("cannot open attribute ")+name),
        H5Aclose
    );
    int value{};
    require_status(H5Aread(attribute.get(),H5T_NATIVE_INT,&value),
                   std::string("cannot read attribute ")+name);
    return value;
}

H5Handle create_group(hid_t parent, const char *name)
{
    return {require_id(H5Gcreate2(parent,name,H5P_DEFAULT,H5P_DEFAULT,
                                  H5P_DEFAULT),
                       std::string("cannot create group ")+name),H5Gclose};
}

H5Handle open_group(hid_t parent, const char *name)
{
    return {require_id(H5Gopen2(parent,name,H5P_DEFAULT),
                       std::string("cannot open group ")+name),H5Gclose};
}

template<class T>
void write_vector(
    hid_t group,
    const char *name,
    const std::vector<T> &values,
    hid_t file_type,
    hid_t memory_type
)
{
    const hsize_t dims[1] = {static_cast<hsize_t>(values.size())};
    H5Handle space(require_id(H5Screate_simple(1,dims,nullptr),
                              std::string("cannot create dataspace for ")+name),
                   H5Sclose);
    H5Handle dataset(
        require_id(H5Dcreate2(group,name,file_type,space.get(),H5P_DEFAULT,
                              H5P_DEFAULT,H5P_DEFAULT),
                   std::string("cannot create dataset ")+name),
        H5Dclose
    );
    if(!values.empty()) {
        require_status(H5Dwrite(dataset.get(),memory_type,H5S_ALL,H5S_ALL,
                                H5P_DEFAULT,values.data()),
                       std::string("cannot write dataset ")+name);
    }
}

template<class T>
std::vector<T> read_vector(hid_t group, const char *name, hid_t memory_type)
{
    H5Handle dataset(require_id(H5Dopen2(group,name,H5P_DEFAULT),
                                std::string("cannot open dataset ")+name),
                     H5Dclose);
    H5Handle space(require_id(H5Dget_space(dataset.get()),
                              std::string("cannot inspect dataset ")+name),
                   H5Sclose);
    if(H5Sget_simple_extent_ndims(space.get())!=1) {
        throw std::runtime_error(std::string(name)+" must be one-dimensional");
    }
    hsize_t dims[1]{};
    H5Sget_simple_extent_dims(space.get(),dims,nullptr);
    std::vector<T> values(static_cast<std::size_t>(dims[0]));
    if(!values.empty()) {
        require_status(H5Dread(dataset.get(),memory_type,H5S_ALL,H5S_ALL,
                               H5P_DEFAULT,values.data()),
                       std::string("cannot read dataset ")+name);
    }
    return values;
}

void write_real_vector(
    hid_t group,
    const char *name,
    const std::vector<Real> &values,
    const char *units
)
{
    std::vector<double> file_values(values.begin(),values.end());
    write_vector(group,name,file_values,H5T_IEEE_F64LE,H5T_NATIVE_DOUBLE);
    H5Handle dataset(require_id(H5Dopen2(group,name,H5P_DEFAULT),
                                std::string("cannot reopen dataset ")+name),
                     H5Dclose);
    write_string_attribute(dataset.get(),"units",units);
}

std::vector<Real> read_real_vector(
    hid_t group,
    const char *name,
    const char *required_units
)
{
    H5Handle dataset(require_id(H5Dopen2(group,name,H5P_DEFAULT),
                                std::string("cannot open dataset ")+name),
                     H5Dclose);
    if(read_string_attribute(dataset.get(),"units")!=required_units) {
        throw std::runtime_error(std::string(name)+" has unsupported units");
    }
    auto values = read_vector<double>(group,name,H5T_NATIVE_DOUBLE);
    return {values.begin(),values.end()};
}

void write_strings(
    hid_t group,
    const char *dataset_name,
    const std::vector<std::string> &names
)
{
    auto type = string_type();
    const hsize_t dims[1] = {static_cast<hsize_t>(names.size())};
    H5Handle space(require_id(H5Screate_simple(1,dims,nullptr),
                              "cannot create string dataspace"),H5Sclose);
    H5Handle dataset(
        require_id(H5Dcreate2(group,dataset_name,type.get(),space.get(),
                              H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT),
                   std::string("cannot create dataset ")+dataset_name),
        H5Dclose
    );
    std::vector<const char *> text;
    text.reserve(names.size());
    for(const auto &name:names) text.push_back(name.c_str());
    if(!text.empty()) {
        require_status(H5Dwrite(dataset.get(),type.get(),H5S_ALL,H5S_ALL,
                                H5P_DEFAULT,text.data()),
                       std::string("cannot write dataset ")+dataset_name);
    }
}

std::vector<std::string> read_strings(hid_t group, const char *dataset_name)
{
    H5Handle dataset(require_id(H5Dopen2(group,dataset_name,H5P_DEFAULT),
                                std::string("cannot open dataset ")+dataset_name),
                     H5Dclose);
    H5Handle space(require_id(H5Dget_space(dataset.get()),
                              "cannot inspect string dataset"),H5Sclose);
    if(H5Sget_simple_extent_ndims(space.get())!=1) {
        throw std::runtime_error(std::string(dataset_name)+
                                 " must be one-dimensional");
    }
    hsize_t dims[1]{};
    H5Sget_simple_extent_dims(space.get(),dims,nullptr);
    auto type = string_type();
    std::vector<char *> raw(static_cast<std::size_t>(dims[0]),nullptr);
    if(!raw.empty()) {
        require_status(H5Dread(dataset.get(),type.get(),H5S_ALL,H5S_ALL,
                               H5P_DEFAULT,raw.data()),
                       std::string("cannot read dataset ")+dataset_name);
    }
    std::vector<std::string> result;
    result.reserve(raw.size());
    for(const char *value:raw) result.emplace_back(value ? value : "");
    if(!raw.empty()) {
        require_status(H5Dvlen_reclaim(type.get(),space.get(),H5P_DEFAULT,
                                       raw.data()),
                       "cannot reclaim variable-length strings");
    }
    return result;
}

void write_complex_matrix(
    hid_t group,
    const char *name,
    const std::vector<Complex> &source,
    std::size_t nrow,
    std::size_t ncolumn,
    const std::string &units
)
{
    std::vector<FileComplex> values;
    values.reserve(source.size());
    for(const Complex value:source) values.push_back({value.real(),value.imag()});
    const hsize_t dims[2] = {
        static_cast<hsize_t>(nrow),static_cast<hsize_t>(ncolumn)
    };
    H5Handle space(require_id(H5Screate_simple(2,dims,nullptr),
                              "cannot create complex dataspace"),H5Sclose);
    auto file_type = complex_type(H5T_IEEE_F64LE);
    auto memory_type = complex_type(H5T_NATIVE_DOUBLE);
    H5Handle dataset(
        require_id(H5Dcreate2(group,name,file_type.get(),space.get(),H5P_DEFAULT,
                              H5P_DEFAULT,H5P_DEFAULT),
                   std::string("cannot create dataset ")+name),
        H5Dclose
    );
    if(!values.empty()) {
        require_status(H5Dwrite(dataset.get(),memory_type.get(),H5S_ALL,H5S_ALL,
                                H5P_DEFAULT,values.data()),
                       std::string("cannot write dataset ")+name);
    }
    write_string_attribute(dataset.get(),"units",units);
}

std::vector<Complex> read_complex_matrix(
    hid_t group,
    const char *name,
    std::size_t expected_rows,
    std::size_t expected_columns,
    std::string &units
)
{
    H5Handle dataset(require_id(H5Dopen2(group,name,H5P_DEFAULT),
                                std::string("cannot open dataset ")+name),
                     H5Dclose);
    units = read_string_attribute(dataset.get(),"units");
    H5Handle space(require_id(H5Dget_space(dataset.get()),
                              "cannot inspect complex dataset"),H5Sclose);
    if(H5Sget_simple_extent_ndims(space.get())!=2) {
        throw std::runtime_error(std::string(name)+" must have rank two");
    }
    hsize_t dims[2]{};
    H5Sget_simple_extent_dims(space.get(),dims,nullptr);
    if(dims[0]!=expected_rows || dims[1]!=expected_columns) {
        throw std::runtime_error(std::string(name)+" has inconsistent shape");
    }
    std::vector<FileComplex> raw(expected_rows*expected_columns);
    auto memory_type = complex_type(H5T_NATIVE_DOUBLE);
    if(!raw.empty()) {
        require_status(H5Dread(dataset.get(),memory_type.get(),H5S_ALL,H5S_ALL,
                               H5P_DEFAULT,raw.data()),
                       std::string("cannot read dataset ")+name);
    }
    std::vector<Complex> result;
    result.reserve(raw.size());
    for(const auto value:raw) result.emplace_back(value.r,value.i);
    return result;
}

void write_real_array3(
    hid_t group,
    const char *name,
    const std::vector<Real> &source,
    std::size_t n0,
    std::size_t n1,
    std::size_t n2,
    const char *units
)
{
    std::vector<double> values(source.begin(),source.end());
    const hsize_t dims[3] = {
        static_cast<hsize_t>(n0),static_cast<hsize_t>(n1),
        static_cast<hsize_t>(n2)
    };
    H5Handle space(require_id(H5Screate_simple(3,dims,nullptr),
                              "cannot create kernel dataspace"),H5Sclose);
    H5Handle dataset(
        require_id(H5Dcreate2(group,name,H5T_IEEE_F64LE,space.get(),H5P_DEFAULT,
                              H5P_DEFAULT,H5P_DEFAULT),
                   std::string("cannot create dataset ")+name),
        H5Dclose
    );
    if(!values.empty()) {
        require_status(H5Dwrite(dataset.get(),H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,
                                H5P_DEFAULT,values.data()),
                       std::string("cannot write dataset ")+name);
    }
    write_string_attribute(dataset.get(),"units",units);
}

void write_csr(
    hid_t parent,
    const char *name,
    const ComplexCSR &csr,
    const std::string &units
)
{
    auto group = create_group(parent,name);
    write_vector(group.get(),"indptr",csr.indptr,H5T_STD_U64LE,
                 H5T_NATIVE_UINT64);
    write_vector(group.get(),"mode_index",csr.mode_index,H5T_STD_I32LE,
                 H5T_NATIVE_INT32);
    write_strings(group.get(),"component_names",csr.component_names);
    write_complex_matrix(group.get(),"values",csr.values,csr.nentry(),
                         csr.ncomponent(),units);
}

ComplexCSR read_csr(hid_t parent, const char *name, std::string &units)
{
    auto group = open_group(parent,name);
    ComplexCSR result;
    result.indptr = read_vector<std::uint64_t>(group.get(),"indptr",
                                               H5T_NATIVE_UINT64);
    result.mode_index = read_vector<std::int32_t>(group.get(),"mode_index",
                                                  H5T_NATIVE_INT32);
    result.component_names = read_strings(group.get(),"component_names");
    result.values = read_complex_matrix(
        group.get(),"values",result.mode_index.size(),
        result.component_names.size(),units
    );
    return result;
}

void write_dispersion(hid_t file, const DispersionTable &table)
{
    write_string_attribute(file,"schema","specswd-dispersion");
    write_integer_attribute(file,"schema_version",1);
    write_string_attribute(file,"layout","csr");
    write_string_attribute(file,"complex_encoding","compound-r-i-f64");

    auto solutions = create_group(file,"/solutions");
    write_real_vector(solutions.get(),"frequency_hz",table.frequency_hz,"Hz");
    write_real_vector(solutions.get(),"azimuth_deg",table.azimuth_deg,"degree");
    write_string_attribute(solutions.get(),"row_order",
                           "frequency-major,azimuth-minor");
    write_csr(solutions.get(),"phase",table.phase_velocity,table.velocity_units);
    if(table.has_group_velocity()) {
        write_csr(solutions.get(),"group",table.group_velocity,
                  table.velocity_units);
    }
}

void write_eigenfunctions(hid_t file, const RaggedComplexField &field)
{
    auto group = create_group(file,"/eigenfunctions");
    write_vector(group.get(),"indptr",field.indptr,H5T_STD_U64LE,
                 H5T_NATIVE_UINT64);
    write_strings(group.get(),"component_names",field.component_names);
    write_complex_matrix(group.get(),"values",field.values,field.npoint(),
                         field.ncomponent(),field.units);
}

void write_kernel_table(
    hid_t parent,
    const char *name,
    const KernelTable &table,
    std::size_t nentry,
    std::size_t ndepth
)
{
    if(table.empty()) return;
    auto group = create_group(parent,name);
    write_strings(group.get(),"parameter_names",table.parameter_names);
    write_real_array3(group.get(),"velocity",table.velocity,nentry,
                      table.parameter_names.size(),ndepth,"kernel");
    if(!table.propagation_q_inverse.empty()) {
        write_real_array3(group.get(),"propagation_q_inverse",
                          table.propagation_q_inverse,nentry,
                          table.parameter_names.size(),ndepth,"kernel");
    }
}

void silence_hdf5_errors()
{
    H5Eset_auto2(H5E_DEFAULT,nullptr,nullptr);
}

void validate_unique_names(
    const std::vector<std::string> &names,
    const std::string &prefix
)
{
    std::unordered_set<std::string> unique;
    for(const auto &name:names) {
        if(name.empty() || !unique.insert(name).second) {
            throw std::invalid_argument(prefix+
                " names must be unique and nonempty");
        }
    }
}

} // namespace

ComplexCSR ComplexCSR::from_rows(
    const std::vector<std::vector<Complex>> &rows,
    std::vector<std::string> names,
    const std::vector<std::vector<std::int32_t>> &mode_rows
)
{
    if(names.empty()) {
        throw std::invalid_argument("component_names must not be empty");
    }
    if(!mode_rows.empty() && mode_rows.size()!=rows.size()) {
        throw std::invalid_argument("mode_rows and rows must have the same size");
    }
    ComplexCSR result;
    result.component_names = std::move(names);
    result.indptr.reserve(rows.size()+1);
    result.indptr.push_back(0);
    for(std::size_t row=0;row<rows.size();++row) {
        if(rows[row].size()%result.ncomponent()!=0) {
            throw std::invalid_argument(
                "CSR row size is not divisible by ncomponent");
        }
        const std::size_t nentry = rows[row].size()/result.ncomponent();
        result.values.insert(result.values.end(),rows[row].begin(),rows[row].end());
        if(mode_rows.empty()) {
            for(std::size_t mode=0;mode<nentry;++mode) {
                if(mode>static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max())) {
                    throw std::overflow_error("mode index exceeds int32 range");
                }
                result.mode_index.push_back(static_cast<std::int32_t>(mode));
            }
        }
        else {
            if(mode_rows[row].size()!=nentry) {
                throw std::invalid_argument("mode row has inconsistent size");
            }
            result.mode_index.insert(result.mode_index.end(),mode_rows[row].begin(),
                                     mode_rows[row].end());
        }
        result.indptr.push_back(result.mode_index.size());
    }
    result.validate(rows.size(),"csr");
    return result;
}

std::size_t ComplexCSR::ncomponent() const noexcept
{
    return component_names.size();
}

std::size_t ComplexCSR::nentry() const noexcept
{
    return mode_index.size();
}

std::pair<std::size_t,std::size_t>
ComplexCSR::row_range(std::size_t row) const
{
    if(row+1>=indptr.size()) throw std::out_of_range("CSR row out of range");
    return {indptr[row],indptr[row+1]};
}

void ComplexCSR::validate(std::size_t nrow, const char *name) const
{
    const std::string prefix(name);
    if(component_names.empty()) {
        throw std::invalid_argument(prefix+" must have at least one component");
    }
    if(indptr.size()!=nrow+1) {
        throw std::invalid_argument(prefix+" indptr must have nsolve+1 entries");
    }
    if(indptr.empty() || indptr.front()!=0) {
        throw std::invalid_argument(prefix+" indptr must start at zero");
    }
    if(!std::is_sorted(indptr.begin(),indptr.end())) {
        throw std::invalid_argument(prefix+" indptr must be nondecreasing");
    }
    if(indptr.back()!=nentry()) {
        throw std::invalid_argument(prefix+" indptr.back() must equal nentry");
    }
    if(values.size()!=nentry()*ncomponent()) {
        throw std::invalid_argument(prefix+" values have inconsistent size");
    }
    validate_unique_names(component_names,prefix+" component");
    for(std::size_t row=0;row<nrow;++row) {
        std::unordered_set<std::int32_t> modes;
        const auto range = row_range(row);
        for(std::size_t i=range.first;i<range.second;++i) {
            if(mode_index[i]<0 || !modes.insert(mode_index[i]).second) {
                throw std::invalid_argument(prefix+
                    " mode indices must be unique and nonnegative per row");
            }
        }
    }
}

std::size_t DispersionTable::nsolve() const noexcept
{
    return frequency_hz.size()*azimuth_deg.size();
}

bool DispersionTable::has_group_velocity() const noexcept
{
    return !group_velocity.indptr.empty();
}

void DispersionTable::validate() const
{
    if(frequency_hz.empty()) {
        throw std::invalid_argument("frequency_hz must not be empty");
    }
    if(azimuth_deg.empty()) {
        throw std::invalid_argument("azimuth_deg must not be empty");
    }
    if(velocity_units.empty()) {
        throw std::invalid_argument("velocity_units must not be empty");
    }
    for(const Real frequency:frequency_hz) {
        if(!std::isfinite(frequency) || frequency<=0.) {
            throw std::invalid_argument("frequencies must be finite and positive");
        }
    }
    for(const Real azimuth:azimuth_deg) {
        if(!std::isfinite(azimuth)) {
            throw std::invalid_argument("azimuths must be finite");
        }
    }
    phase_velocity.validate(nsolve(),"phase_velocity");
    if(phase_velocity.component_names!=std::vector<std::string>{"phase"}) {
        throw std::invalid_argument(
            "phase_velocity must have the single component 'phase'");
    }
    if(has_group_velocity()) group_velocity.validate(nsolve(),"group_velocity");
}

std::size_t RaggedComplexField::ncomponent() const noexcept
{
    return component_names.size();
}

std::size_t RaggedComplexField::npoint() const noexcept
{
    return ncomponent()==0 ? 0 : values.size()/ncomponent();
}

void RaggedComplexField::validate(std::size_t nrow, const char *name) const
{
    const std::string prefix(name);
    if(component_names.empty()) {
        throw std::invalid_argument(prefix+" component names must not be empty");
    }
    validate_unique_names(component_names,prefix+" component");
    if(units.empty()) throw std::invalid_argument(prefix+" units must not be empty");
    if(indptr.size()!=nrow+1 || indptr.empty() || indptr.front()!=0 ||
       !std::is_sorted(indptr.begin(),indptr.end())) {
        throw std::invalid_argument(prefix+
            " indptr must contain nrow+1 nondecreasing entries starting at zero");
    }
    if(values.size()%ncomponent()!=0 || indptr.back()!=npoint()) {
        throw std::invalid_argument(prefix+" values have inconsistent size");
    }
}

bool KernelTable::empty() const noexcept
{
    return parameter_names.empty();
}

void KernelTable::validate(
    std::size_t nentry,
    std::size_t ndepth,
    bool attenuation,
    const char *name
) const
{
    const std::string prefix(name);
    if(empty()) {
        if(!velocity.empty() || !propagation_q_inverse.empty()) {
            throw std::invalid_argument(prefix+" has values but no parameter names");
        }
        return;
    }
    validate_unique_names(parameter_names,prefix+" parameter");
    const std::size_t required = nentry*parameter_names.size()*ndepth;
    if(velocity.size()!=required) {
        throw std::invalid_argument(prefix+" velocity values have inconsistent size");
    }
    if(attenuation) {
        if(propagation_q_inverse.size()!=required) {
            throw std::invalid_argument(prefix+
                " propagation-Q-inverse values have inconsistent size");
        }
    }
    else if(!propagation_q_inverse.empty()) {
        throw std::invalid_argument(prefix+
            " propagation-Q-inverse values require attenuation");
    }
}

void SolverOutput::validate() const
{
    dispersion.validate();
    if(wave_type<0 || wave_type>2) {
        throw std::invalid_argument("wave_type must be 0, 1, or 2");
    }
    if(kernel_type!=0 && kernel_type!=1) {
        throw std::invalid_argument("kernel_type must be 0 or 1");
    }
    if(kernel_observable.empty()) {
        throw std::invalid_argument("kernel_observable must not be empty");
    }
    if(model_depth.empty()) {
        throw std::invalid_argument("model_depth must not be empty");
    }
    if(sem_depth_indptr.size()!=dispersion.nsolve()+1 ||
       sem_depth_indptr.empty() || sem_depth_indptr.front()!=0 ||
       !std::is_sorted(sem_depth_indptr.begin(),sem_depth_indptr.end()) ||
       sem_depth_indptr.back()!=sem_depth_values.size()) {
        throw std::invalid_argument("SEM depth CSR has inconsistent size");
    }
    eigenfunctions.validate(dispersion.phase_velocity.nentry(),"eigenfunctions");
    std::size_t entry = 0;
    for(std::size_t row=0;row<dispersion.nsolve();++row) {
        const std::size_t ndepth = sem_depth_indptr[row+1]-sem_depth_indptr[row];
        const auto range = dispersion.phase_velocity.row_range(row);
        for(entry=range.first;entry<range.second;++entry) {
            if(eigenfunctions.indptr[entry+1]-eigenfunctions.indptr[entry]
               !=ndepth) {
                throw std::invalid_argument(
                    "eigenfunction length does not match its SEM depth row");
            }
        }
    }
    const auto nmode = dispersion.phase_velocity.nentry();
    elastic_kernels.validate(nmode,model_depth.size(),attenuation,
                             "elastic_kernels");
    acoustic_kernels.validate(nmode,model_depth.size(),attenuation,
                              "acoustic_kernels");
}

void write_hdf5(const std::string &path, const DispersionTable &table)
{
    table.validate();
    silence_hdf5_errors();
    try {
        H5Handle file(require_id(H5Fcreate(path.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,
                                           H5P_DEFAULT),
                                 "cannot create HDF5 file"),H5Fclose);
        write_dispersion(file.get(),table);
    }
    catch(const std::exception &error) {
        throw std::runtime_error("failed to write HDF5 file '"+path+"': "+
                                 error.what());
    }
}

void write_hdf5(const std::string &path, const SolverOutput &output)
{
    output.validate();
    silence_hdf5_errors();
    try {
        H5Handle file(require_id(H5Fcreate(path.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,
                                           H5P_DEFAULT),
                                 "cannot create HDF5 file"),H5Fclose);
        write_dispersion(file.get(),output.dispersion);
        write_string_attribute(file.get(),"content","solver-output");
        write_integer_attribute(file.get(),"wave_type",output.wave_type);
        write_integer_attribute(file.get(),"attenuation",output.attenuation ? 1:0);
        write_integer_attribute(file.get(),"kernel_type",output.kernel_type);

        auto model = create_group(file.get(),"/model");
        write_real_vector(model.get(),"depth",output.model_depth,"km");

        auto mesh = create_group(file.get(),"/mesh");
        write_vector(mesh.get(),"depth_indptr",output.sem_depth_indptr,
                     H5T_STD_U64LE,H5T_NATIVE_UINT64);
        write_real_vector(mesh.get(),"depth_values",output.sem_depth_values,"km");
        write_string_attribute(mesh.get(),"row_alignment","solutions");

        write_eigenfunctions(file.get(),output.eigenfunctions);
        auto kernels = create_group(file.get(),"/kernels");
        write_string_attribute(kernels.get(),"observable",output.kernel_observable);
        write_string_attribute(kernels.get(),"entry_alignment","solutions/phase");
        write_kernel_table(kernels.get(),"elastic",output.elastic_kernels,
                           output.dispersion.phase_velocity.nentry(),
                           output.model_depth.size());
        write_kernel_table(kernels.get(),"acoustic",output.acoustic_kernels,
                           output.dispersion.phase_velocity.nentry(),
                           output.model_depth.size());
    }
    catch(const std::exception &error) {
        throw std::runtime_error("failed to write HDF5 file '"+path+"': "+
                                 error.what());
    }
}

DispersionTable read_hdf5(const std::string &path)
{
    silence_hdf5_errors();
    try {
        H5Handle file(require_id(H5Fopen(path.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),
                                 "cannot open HDF5 file"),H5Fclose);
        if(read_string_attribute(file.get(),"schema")!="specswd-dispersion" ||
           read_integer_attribute(file.get(),"schema_version")!=1 ||
           read_string_attribute(file.get(),"layout")!="csr" ||
           read_string_attribute(file.get(),"complex_encoding")!=
               "compound-r-i-f64") {
            throw std::runtime_error("unsupported SpecSWD HDF5 schema");
        }
        auto solutions = open_group(file.get(),"/solutions");
        if(read_string_attribute(solutions.get(),"row_order")!=
           "frequency-major,azimuth-minor") {
            throw std::runtime_error("unsupported solution row order");
        }
        DispersionTable result;
        result.frequency_hz = read_real_vector(solutions.get(),"frequency_hz","Hz");
        result.azimuth_deg = read_real_vector(solutions.get(),"azimuth_deg","degree");
        result.phase_velocity = read_csr(solutions.get(),"phase",
                                         result.velocity_units);
        if(H5Lexists(solutions.get(),"group",H5P_DEFAULT)>0) {
            std::string group_units;
            result.group_velocity = read_csr(solutions.get(),"group",group_units);
            if(group_units!=result.velocity_units) {
                throw std::runtime_error(
                    "phase and group velocity units must match");
            }
        }
        result.validate();
        return result;
    }
    catch(const std::exception &error) {
        throw std::runtime_error("failed to read HDF5 file '"+path+"': "+
                                 error.what());
    }
}

} // namespace specswd::io
