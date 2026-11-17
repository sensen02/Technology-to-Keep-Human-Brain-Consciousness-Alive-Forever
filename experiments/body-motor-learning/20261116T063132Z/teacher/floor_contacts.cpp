#include <algorithm>
#include <cmath>
#include <mujoco/mujoco.h>

// Ground roles are read from the compiled static floor layout, shared for all body parts.
extern "C" void gap_totals(const mjModel* model, const mjData* data, const int* feet,
                          const int* role, int probe_leg, const double* velocity,
                          double* result) {
    std::fill(result, result+17, 0.);
    double jac[3*48];
    double load[6];
    for (int i=0; i<data->ncon; ++i) {
        const auto& c=data->contact[i];
        int a=c.geom[0], b=c.geom[1];
        if (a<0 || b<0 || c.efc_address<0 || (bool(role[a])==bool(role[b]))) continue;
        int floor=role[a] ? a : b;
        int geom=role[a] ? b : a;
        int leg=feet[geom];
        mj_contactForce(model,data,i,load);
        double orientation=role[a] ? 1. : -1.;
        double normal=orientation*c.frame[2];
        double support=std::max(0.,orientation*(c.frame[2]*load[0]+c.frame[5]*load[1]+c.frame[8]*load[2]));
        if (leg<0) {result[12]+=support; continue;}
        result[leg]+=support;
        mj_jac(model,data,jac,nullptr,c.pos,model->geom_bodyid[geom]);
        double vx=0.,vy=0.;
        for(int j=0;j<model->nv;++j) {
            vx+=jac[j]*velocity[j]; vy+=jac[model->nv+j]*velocity[j];
        }
        result[leg+6]+=support*std::hypot(vx,vy);
        if (leg==probe_leg) {
            if (normal>.8) result[role[floor]==2 ? 13 : 14]+=support;
            else {result[15]+=support; result[16]+=load[0];}
        }
    }
}
