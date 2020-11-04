// Single-seizure model for seizure propagation inference
// Viktor Sip, 2020

functions {
    vector expbilin(vector y, vector c, real q11, real q12, real q21, real q22) {
        real c1 = -1;
        real c2 = 1;
        real y1 = 0;
        real y2 = 1;

        return exp(1./((c2 - c1)*(y2 - y1)) * (  q11*(c2 - c) .* (y2 - y)
                                               + q21*(c - c1) .* (y2 - y)
                                               + q12*(c2 - c) .* (y - y1)
                                               + q22*(c - c1) .* (y - y1)));
    }

    real[] prop(int nreg, vector c, matrix w, real q11, real q12, real q21, real q22) {
        real t[nreg];
        vector[nreg] x;
        vector[nreg] z;
        vector[nreg] y;
        vector[nreg] fy;
        real time;

        int reg_to_switch;
        real dt;
        real dt_;

        x = rep_vector(0.0, nreg);
        z = rep_vector(0.0, nreg);

        time = 0;
        for (k in 1:nreg) {
            y = w * x;
            fy = expbilin(y, c, q11, q12, q21, q22);

            // find first region that will switch
            dt = positive_infinity();

            for (i in 1:nreg) {
                if (x[i] < 0.5) {
                    dt_ = (1.0 - z[i])/fy[i];
                    if (dt_ < dt) {
                        dt = dt_;
                        reg_to_switch = i;
                    }
                }
            }

            z += dt * fy;
            time += dt;
            x[reg_to_switch] = 1;
            t[reg_to_switch] = time;
        }

        return t;
    }
}


data {
    int nreg;                           // Number of regions
    matrix<lower=0.0>[nreg, nreg] w;    // Structural connectome

    int n_sz;                           // Number of seizing regions
    int reg_sz[n_sz];                   // Seizing regions (0-based indexing) 
    vector[n_sz] t_sz;                  // Onset times of the seizing regions

    int n_ns;                           // Number of non-seizing regions
    int reg_ns[n_ns];                   // Non-seizing regions (0-based indexing)

    real t_lim;                         // Limit time
    real sig_t;                         // sigma_t 

    real q11;                           // Hyperparameters
    real q12;
    real<lower=0.0> qa21;
    real<lower=0.0> qa22;
}

transformed data {
    int ireg_sz[n_sz];
    int ireg_ns[n_ns];
    vector[4] q;

    // Adapt to 1-based indexing in Stan
    for (i in 1:n_sz) {
        ireg_sz[i] = reg_sz[i] + 1;
    }
    for (i in 1:n_ns) {
        ireg_ns[i] = reg_ns[i] + 1;
    }

    q[1] = q11;
    q[2] = q12;
    q[3] = q11 + qa21;
    q[4] = q12 + qa22;
}

parameters {
    vector[nreg] c;                     // Excitabilities
}


transformed parameters {

}


model {
    real t[nreg];                       // Onset times

    c ~ normal(0, 1);
    t = prop(nreg, c, w, q[1], q[2], q[3], q[4]);

    // Seizing nodes
    for (i in 1:n_sz) {
        t_sz[i] ~ normal(t[ireg_sz[i]], sig_t);
    }
    
    // Non-seizing nodes
    for (i in 1:n_ns) {
        t_lim ~ normal(fmin(t_lim, t[ireg_ns[i]]), sig_t);
    }
}


generated quantities {
    real t[nreg];

    t = prop(nreg, c, w, q[1], q[2], q[3], q[4]);
}
