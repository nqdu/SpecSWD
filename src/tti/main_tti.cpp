#include "tti/swdlayertti.hpp"
#include "swdio.hpp"

#include <iostream>
#include <fstream>

int main (int argc, char **argv){
    // read model name
    if(argc != 6 & argc != 7) {
        printf("Usage: ./surfvti modelfile phi f1 f2 nt [is_layered=1]\n");
        printf("freqs = logspace(log10(f1),log10(f2),nt)\n");
        exit(1);
    }

    // read flag if required
    bool is_layer = true;
    if(argc == 7) {
        int flag;
        sscanf(argv[6],"%d",&flag);
        is_layer = (flag == 1);
    }

    // read model
    std::ifstream infile; infile.open(argv[1]);
    if(is_layer) {
        printf("reading layered velocity model:\n");
    }
    else {
        printf("reading continuous velocity model:\n");
    }
    printf("layer number\t thick\t rho\t vpv\t vph\t vsv\t vsh\t eta\t theta\t phi  \n");
    std::vector<float> thk,vpv,vph,vsv,vsh,rho,theta0,phi0,eta;
    int nz;
    infile >> nz; 
    thk.resize(nz); vsv.resize(nz); vsh.resize(nz); rho.resize(nz);
    vpv.resize(nz); vph.resize(nz); eta.resize(nz); theta0.resize(nz);
    phi0.resize(nz);
    for(int i = 0; i < nz; i ++) {
        infile >> thk[i] >> rho[i] >> vpv[i] >>
                  vph[i] >> vsv[i] >> vsh[i] >>
                  eta[i] >> theta0[i] >> phi0[i];
        printf("layer %d\t %g\t %g\t %g\t %g\t %g\t %g\t %g\t %g\t %g\n",
                i + 1,thk[i],rho[i],vpv[i],
               vph[i],vsv[i],vsh[i],eta[i],
               theta0[i],phi0[i]);
        theta0[i] *= M_PI / 180.;
        phi0[i] *= M_PI / 180.;
    }
    infile.close();

    // Period
    int nt;
    float f1,f2;
    sscanf(argv[3],"%g",&f1); sscanf(argv[4],"%g",&f2);
    sscanf(argv[5],"%d",&nt);
    f1 = std::log10(f1); f2 = std::log10(f2);
    if(f1 > f2) std::swap(f1,f2);
    std::vector<double> freq(nt);
    for(int it = 0; it < nt; it ++) {
        double coef = (nt - 1);
        if(coef == 0.) coef = 1.;
        coef = 1. / coef;
        double f = f1 + (f2 - f1) * coef * it;
        freq[it] = std::pow(10,f);
    }

    // angle
    float phi;
    sscanf(argv[2],"%f",&phi);

    // create database
    printf("\ncomputing dispersions for TTI model, angle = %f ...\n",phi);
    LayerModelTTI model;
    model.initialize();

    FILE *fp = fopen("out/swd.txt","w");
    FILE *fio = fopen("out/database.bin","wb");

    // write period vector into fp
    for(int it = 0; it < nt; it ++) {
        fprintf(fp,"%g ",1. / freq[it]);
    }
    fprintf(fp,"\n");

    // write meta info into database
    int nkers = 8, ncomp = 3;
    write_binary_f(fio,&nkers,1);
    write_binary_f(fio,&ncomp,1);

    for(int it = 0; it < nt; it ++) {
        std::vector<double> c;
        std::vector<std::complex<double>> displ;

        // phase velocity/eigenfunctions
        model.create_database(
            freq[it],nz,rho.data(),vpv.data(),vph.data(),vsv.data(),
            vsh.data(),eta.data(),theta0.data(),phi0.data(),
            thk.data(),is_layer);
        model.prepare_matrices(phi);
        model.compute_egnfun(freq[it],phi,c,displ);

        // get some consts
        int nglob = model.nglob;
        int npts = model.ibool.size();

        // write coordinates
        write_binary_f(fio,model.znodes.data(),npts);

        // group
        std::vector<double> frekl;
        int nc = c.size();
        double u,uphi;
        for(int ic = 0; ic < nc; ic ++) {
            auto out = model.compute_kernels(freq[it],c[ic],phi,&displ[ic*nglob*3],frekl);
            u = out[0]; uphi = out[1];
            model.transform_kernels(frekl);

            // write swd
            fprintf(fp,"%d %g %g %g %g %d\n",it,c[ic],u,phi,uphi,ic);

            // write displ
            std::vector<std::complex<double>> temp(npts*ncomp);
            for(int i = 0; i < npts; i ++) {
                int iglob = model.ibool[i];
                for(int j = 0; j < ncomp; j ++) {
                    temp[j * npts + i] = displ[ic * nglob * ncomp + j * nglob + iglob];
                }
            }
            write_binary_f(fio,temp.data(),npts*ncomp);

            // write kernels
            write_binary_f(fio,&frekl[0],npts*nkers);
        }
    }
    fclose(fio);
    fclose(fp);
    
}