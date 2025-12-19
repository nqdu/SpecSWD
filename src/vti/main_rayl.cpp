#include "vti/vti.hpp"
#include "shared/iofunc.hpp"
#include "shared/GQTable.hpp"

#include <iostream>
#include <filesystem>
#include <memory>


int main (int argc, char **argv){
    // read model name
    if(argc != 5 &&  argc != 6) {
        printf("Usage: ./surfrayl modelfile f1 f2 nt [KERNEL_TYPE = 0]\n");
        printf("freqs = logspace(log10(f1),log10(f2),nt)\n");
        exit(1);
    }

    // initialize GLL
    GQTable:: initialize();
    using specswd::real_t;

    // read mesh 
    const char *filename = argv[1];
    auto mesh = std::make_unique<specswd::Mesh>();
    mesh->read_model(filename);
    mesh->create_model_attributes();
    int nz = mesh->nz_tomo;

    // check if it's Rayleigh wave
    if(mesh->SWD_TYPE != 1) {
        printf("THis module can only handle Rayleigh wave!\n");
        exit(1);
    }

    // print info to debug
    mesh->print_model();

    // Period
    int nt;
    float f1,f2;
    sscanf(argv[2],"%g",&f1); sscanf(argv[3],"%g",&f2);
    sscanf(argv[4],"%d",&nt);
    f1 = std::log10(f1); f2 = std::log10(f2);
    if(f1 > f2) std::swap(f1,f2);
    std::vector<double> freq(nt);
    for(int it = 0; it < nt; it ++) {
        double coef = (nt - 1);
        coef = coef == 0 ? 1: coef;
        coef = 1. / coef;
        double f = f1 + (f2 - f1) * coef * it;
        freq[it] = std::pow(10,f);
    }

    int KERNEL_TYPE = 0;
    if(argc == 6) {
        sscanf(argv[5],"%d",&KERNEL_TYPE);
    }

    // create output dir
    if(!std::filesystem::exists("out/"))
        std::filesystem::create_directory("out/");
    
    // open file to write out meta data
    FILE *fp = fopen("out/swd.txt","w");
    FILE *fio = fopen("out/database.bin","wb");
    for(int it = 0; it < nt; it ++) {
        fprintf(fp,"%g ",1. / freq[it]);
    }
    fprintf(fp,"\n");

    // create solver
    auto sol = std::make_unique<specswd::SolverRayl>();
    sol->build(mesh.get());

    // write meta data int database
    using specswd::write_binary_f;
    int ncomp = 2;
    write_binary_f(fio,&mesh->SWD_TYPE,1);
    write_binary_f(fio,&mesh->HAS_ATT,1);
    write_binary_f(fio,&nz,1);
    int nkers_ac = sol->nkers_ac;
    int nkers_el = sol->nkers_el;
    int nkers = nkers_el + nkers_ac;
    write_binary_f(fio,&nkers,1);
    write_binary_f(fio,&ncomp,1);

    // compute phase velocity for each frequency
    for(int it = 0; it < nt; it ++) {
        // create database
        mesh->create_database(freq[it],0.);

        if(it == 0) mesh->print_database();

        // write coordinates
        std::vector<real_t> zcoord = mesh->znodes;
        for(auto &zz : zcoord) {
            zz *= mesh->SCALE_LENGTH;
        }
        write_binary_f(fio,zcoord.data(),zcoord.size());

        // prepare all matrices
        sol -> prepare_matrices();

        // compute eigenvalue
        sol -> compute_egn(true);

        // compute group velocity
        sol-> compute_group_vel();

        if(!mesh->HAS_ATT) {
            std::vector<real_t> c,ur,ul,u;
            std::vector<real_t> frekl_el,frekl_ac;
            std::vector<real_t> frekl_el_tmp,frekl_ac_tmp;
            std::vector<real_t> frekl_tomo;
            std::vector<real_t> displ;
            std::vector<specswd::complex_t> displ_tmp;

            // allocate group velocity
            int nc = sol->c_phase.size();
            c.resize(nc);
            u.resize(nc);

            for(int ic = 0; ic < nc; ic ++) {
                real_t temp;
                sol->get_phase_vel(ic,c[ic],temp);
                sol->get_group_vel(ic,u[ic],temp);
                sol->compute_kernels(
                    ic,
                    KERNEL_TYPE,
                    frekl_el,
                    frekl_el_tmp,
                    frekl_ac,
                    frekl_ac_tmp
                );

                // write T,c,u,mode
                fprintf(fp,"%d %g %g %d\n",it,c[ic],u[ic],ic);

                // write displ
                displ.resize(mesh->ibool.size() * 2);
                displ_tmp.resize(displ.size());
                sol->egn2displ(ic, displ_tmp.data());
                for(size_t i = 0; i < displ.size(); i ++) {
                    displ[i] = displ_tmp[i].real();
                }
                write_binary_f(fio,displ.data(),displ.size());

                // transform elastic kernels 
                int npts = mesh->ibool.size();
                frekl_tomo.resize(nkers_el*nz);
                
                for(int iker = 0; iker < nkers_el; iker ++) {
                    mesh->project_kl(&frekl_el[iker*npts],&frekl_tomo[iker*nz]);
                }
                write_binary_f(fio,frekl_tomo.data(),frekl_tomo.size());

                // transform acoustic kernels
                frekl_tomo.resize(nkers_ac*nz);
                for(int iker = 0; iker < nkers_ac; iker ++) {
                    mesh->project_kl(&frekl_ac[iker*npts],&frekl_tomo[iker*nz]);
                }
                write_binary_f(fio,frekl_tomo.data(),frekl_tomo.size());
            }
        }
        else {
            using specswd::complex_t;
            std::vector<complex_t> c,ur,ul,u;
            std::vector<real_t> frekl_el_c,frekl_el_q;
            std::vector<real_t> frekl_ac_c,frekl_ac_q;
            std::vector<real_t> frekl_tomo;
            std::vector<complex_t> displ;

            // allocate group velocity
            int nc = sol->c_phase.size();
            c.resize(nc);
            u.resize(nc);

            for(int ic = 0; ic < nc; ic ++) {
                real_t val_r,val_i;
                sol->get_phase_vel(ic,val_r,val_i);
                c[ic] = complex_t{val_r,val_i};
                sol->get_group_vel(ic,val_r,val_i);
                u[ic] = complex_t{val_r,val_i};

                sol->compute_kernels(
                    ic,
                    KERNEL_TYPE,
                    frekl_el_c,
                    frekl_el_q,
                    frekl_ac_c,
                    frekl_ac_q
                );

                // write T,c,u,mode
                fprintf(fp,"%d %g %g %g %g %d\n",it,c[ic].real(),u[ic].real(),
                                                c[ic].imag(),u[ic].imag(),ic);

                // write displ
                int npts = mesh->ibool.size();
                displ.resize(npts * 2);
                sol -> egn2displ(ic,displ.data());
                write_binary_f(fio,displ.data(),displ.size());

                // transform elastic kernels
                frekl_tomo.resize(nkers_el*nz);
                for(int iker = 0; iker < nkers_el; iker ++) {
                    mesh->project_kl(&frekl_el_c[iker*npts],&frekl_tomo[iker*nz]);
                }
                write_binary_f(fio,frekl_tomo.data(),frekl_tomo.size());

                for(int iker = 0; iker < nkers_el; iker ++) {
                    mesh->project_kl(&frekl_el_q[iker*npts],&frekl_tomo[iker*nz]);
                }
                write_binary_f(fio,frekl_tomo.data(),frekl_tomo.size());

                // transform acoustic kernels
                frekl_tomo.resize(nkers_ac*nz);
                for(int iker = 0; iker < nkers_ac; iker ++) {
                    mesh->project_kl(&frekl_ac_c[iker*npts],&frekl_tomo[iker*nz]);
                }
                write_binary_f(fio,frekl_tomo.data(),frekl_tomo.size());

                for(int iker = 0; iker < nkers_ac; iker ++) {
                    mesh->project_kl(&frekl_ac_q[iker*npts],&frekl_tomo[iker*nz]);
                }
                write_binary_f(fio,frekl_tomo.data(),frekl_tomo.size());
            }
        }
    }

    // close file
    fclose(fio);
    fclose(fp);
    
    return 0;
}