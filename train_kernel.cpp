#include <cmath>
#include <cstdint>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

// Parameters are [w1,b1,w2,b2,v1,v2]. All six gradients use the old iterate.
extern "C" void train_batch(const double* initial, const double* lr,
    const int* train_output, int n, const int64_t* checkpoints, int k,
    int64_t filter_step, double filter_loss, int threads,
    double* snapshots, int* status, int64_t* stop_step) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    #pragma omp parallel for schedule(dynamic,1)
    for (int run=0; run<n; ++run) {
        double w1=initial[6*run], b1=initial[6*run+1];
        double w2=initial[6*run+2], b2=initial[6*run+3];
        double v1=initial[6*run+4], v2=initial[6*run+5];
        const double eta=lr[run];
        int next=0;
        status[run]=0;
        for (int64_t t=0; t<=checkpoints[k-1]; ++t) {
            if (next<k && t==checkpoints[next]) {
                double* dst=snapshots+((int64_t)run*k+next)*6;
                dst[0]=w1; dst[1]=b1; dst[2]=w2; dst[3]=b2; dst[4]=v1; dst[5]=v2;
                ++next;
            }
            const double zm1=-w1+b1, zp1=w1+b1;
            const double zm2=-w2+b2, zp2=w2+b2;
            const double rm1=std::max(zm1,0.0), rp1=std::max(zp1,0.0);
            const double rm2=std::max(zm2,0.0), rp2=std::max(zp2,0.0);
            const double fm=v1*rm1+v2*rm2, fp=v1*rp1+v2*rp2;
            const double em=std::exp(fm), ep=std::exp(-fp);
            if (!std::isfinite(em) || !std::isfinite(ep)) {
                status[run]=2; stop_step[run]=t; break;
            }
            if (t==filter_step && 0.5*(em+ep)>=filter_loss) {
                status[run]=1; stop_step[run]=t; break;
            }
            if (t==checkpoints[k-1]) {stop_step[run]=t; break;}
            const double cm=0.5*em, cp=-0.5*ep;
            const double gm1=zm1>0 ? cm : 0.0, gp1=zp1>0 ? cp : 0.0;
            const double gm2=zm2>0 ? cm : 0.0, gp2=zp2>0 ? cp : 0.0;
            const double dw1=v1*(-gm1+gp1), db1=v1*(gm1+gp1);
            const double dw2=v2*(-gm2+gp2), db2=v2*(gm2+gp2);
            const double dv1=cm*rm1+cp*rp1, dv2=cm*rm2+cp*rp2;
            w1-=eta*dw1; b1-=eta*db1; w2-=eta*dw2; b2-=eta*db2;
            if (train_output[run]) {v1-=eta*dv1; v2-=eta*dv2;}
        }
    }
}
