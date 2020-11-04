// Multi-seizure model for seizure propagation inference
// Viktor Sip, 2020

// The code mirrors the structure of the single-seizure inference, except that it
// processess multiple seizures at once and hyperparameters q are parameters and
// not constants. Some additional complexity is due to the parallelization with 
// map_rect and the associated data packing.


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


    vector simprop(vector q, vector c, real[] x_r, int[] x_i) {
        int nreg = num_elements(c);

        // Unpack data
        int s = 3 + nreg*nreg;
        real t_lim = x_r[1];
        real sig_t = x_r[2];
        vector[nreg] t_sz = to_vector(x_r[s:s+nreg-1]);
        int n_sz = x_i[1];
        int n_ns = x_i[2];
        int ireg_sz[nreg] = x_i[3:3+nreg-1];
        int ireg_ns[nreg] = x_i[3+nreg:3+2*nreg-1];

        real t[nreg];
        real lp = 0;

        // Calculate the logpdf
        lp += normal_lpdf(c | 0, 1);

        t = prop(nreg, c, to_matrix(x_r[3:s-1], nreg, nreg), q[1], q[2], q[3], q[4]);

        for (i in 1:n_sz) {
            lp += normal_lpdf(t_sz[i] | t[ireg_sz[i]], sig_t);
        }
        for (i in 1:n_ns) {
            lp += normal_lpdf(t_lim | fmin(t_lim, t[ireg_ns[i]]), sig_t);
        }

        return [lp]';
    }

    real[,] pack_data_r(int ns, int nreg, real t_lim, real sig_t, matrix[] w, vector[] t_sz) {
        // Group all real data
        
        real x_r[ns, 2 + nreg*nreg + nreg];
        int s = 3 + nreg*nreg;
        for (i in 1:ns) {
            x_r[i, 1] = t_lim;
            x_r[i, 2] = sig_t;
            x_r[i, 3:s-1] = to_array_1d(w[i]);
            x_r[i, s:s+nreg-1] = to_array_1d(t_sz[i]);
        }
        return x_r;
    }

    int[,] pack_data_i(int ns, int nreg, int[] n_sz, int[] n_ns, int[,] ireg_sz, int[,] ireg_ns) {
        // Group all integer data
        
        int x_i[ns, 2 + 2*nreg];
        for (i in 1:ns) {
            x_i[i, 1] = n_sz[i];
            x_i[i, 2] = n_ns[i];
            x_i[i, 3:3+nreg-1] = ireg_sz[i];
            x_i[i, 3+nreg:3+2*nreg-1] = ireg_ns[i];
        }
        return x_i;
    }
}


data {
    int n_seizures;                                 // Number of seizures

    int nreg;                                       // Number of regions
    matrix<lower=0.0>[nreg, nreg] w[n_seizures];    // Structural connectomes (for all seizures)

    int n_sz[n_seizures];                           // Numbers of seizing regions
    int reg_sz[n_seizures, nreg];                   // Seizing regions (0-based indexing)
    vector[nreg] t_sz[n_seizures];                  // Onset times of seizing regions

    int n_ns[n_seizures];                           // Number of non-seizing regions
    int reg_ns[n_seizures, nreg];                   // Non-seizing regions (0-based indexing)

    real t_lim;                                     // Limit time
    real sig_t;                                     // sigma_t
}

transformed data {
    int ireg_sz[n_seizures, nreg];
    int ireg_ns[n_seizures, nreg];
    real x_r[n_seizures, 2 + nreg*nreg + nreg];
    int  x_i[n_seizures, 2 + 2*nreg];

    // Adapt to 1-based indexing in Stan
    for (i in 1:n_seizures) {
        for (j in 1:nreg) {
            ireg_sz[i, j] = reg_sz[i, j] + 1;
            ireg_ns[i, j] = reg_ns[i, j] + 1;
        }
    }

    x_r = pack_data_r(n_seizures, nreg, t_lim, sig_t, w, t_sz);
    x_i = pack_data_i(n_seizures, nreg, n_sz, n_ns, ireg_sz, ireg_ns);
}

parameters {
    real q11;                     // Hyperparameters
    real q12;
    real<lower=0.0> qa21;
    real<lower=0.0> qa22;

    vector[nreg] c[n_seizures];   // Excitabilities
}

transformed parameters {

}


model {
    vector[4] q;

    q11 ~ normal(0, 30);
    q12 ~ normal(0, 30);
    qa21 ~ normal(0, 30);
    qa22 ~ normal(0, 30);

    q[1] = q11;
    q[2] = q12;
    q[3] = q11 + qa21;
    q[4] = q12 + qa22;

    target += sum(map_rect(simprop, q, c, x_r, x_i));
}


generated quantities {
    real t[n_seizures, nreg];
    real q21;
    real q22;

    q21 = q11 + qa21;
    q22 = q12 + qa22;

    for (k in 1:n_seizures) {
        t[k] = prop(nreg, c[k], w[k], q11, q12, q21, q22);
    }
}
