"""GRU-ODE-Bayes core, vendored unchanged apart from removed debugger calls.

Source : https://github.com/edebrouwer/gru_ode_bayes (gru_ode_bayes/models.py)
Paper  : De Brouwer, Simm, Arany, Moreau. GRU-ODE-Bayes: Continuous modeling of
         sporadically-observed time series. NeurIPS 2019.
License: MIT (Copyright (c) 2019 Edward De Brouwer). Only the classes this project
         uses are kept. Use solver="euler" or "midpoint": the repo's "dopri5" path
         needs its patched torchdiffeq and is not supported here.
"""
import math

import numpy as np
import torch

class GRUODECell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias        = bias

        self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, x, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            x        input values
            h        hidden state (current)
            delta_t  time step

        Returns:
            Updated h
        """
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        n = torch.tanh(self.lin_xn(x) + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh

class GRUObservationCellLogvar(torch.nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super().__init__()
        self.gru_d     = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gru_debug = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        ## prep layer and its initialization
        std            = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep    = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size  = input_size
        self.prep_hidden = prep_hidden

    def forward(self, h, p, X_obs, M_obs, i_obs):
        ## only updating rows that have observations
        p_obs        = p[i_obs]

        mean, logvar = torch.chunk(p_obs, 2, dim=1)
        sigma        = torch.exp(0.5 * logvar)
        error        = (X_obs - mean) / sigma

        ## log normal loss, over all observations
        log_lik_c    = np.log(np.sqrt(2*np.pi))
        losses       = 0.5 * ((torch.pow(error, 2) + logvar + 2*log_lik_c) * M_obs)

        ## TODO: try removing X_obs (they are included in error)
        gru_input    = torch.stack([X_obs, mean, logvar, error], dim=2).unsqueeze(2)
        gru_input    = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is (sample x feature x prep_hidden)
        gru_input    = gru_input.permute(2, 0, 1)
        gru_input    = (gru_input * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h, losses

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)

class NNFOwithBayesianJumps(torch.nn.Module):
    ## Neural Negative Feedback ODE with Bayesian jumps
    def __init__(self, input_size, hidden_size, p_hidden, prep_hidden, bias=True, cov_size=1, cov_hidden=1, classification_hidden=1, logvar=True, mixing=1, dropout_rate=0, full_gru_ode=False, solver="euler", impute = True, **options):
        """
        The smoother variable computes the classification loss as a weighted average of the projection of the latents at each observation.
        impute feeds the parameters of the distribution to GRU-ODE at each step.
        """

        super().__init__()

        self.impute = impute
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )

        self.classification_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size,classification_hidden,bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(classification_hidden,1,bias=bias)
        )
        if full_gru_ode:
            if impute is False:
                self.gru_c   = FullGRUODECell_Autonomous(hidden_size, bias = bias)
            else:
                self.gru_c   = FullGRUODECell(2 * input_size, hidden_size, bias=bias)

        else:
            if impute is False:
                self.gru_c = GRUODECell_Autonomous(hidden_size, bias= bias)
            else:
                self.gru_c   = GRUODECell(2 * input_size, hidden_size, bias=bias)

        if logvar:
            self.gru_obs = GRUObservationCellLogvar(input_size, hidden_size, prep_hidden, bias=bias)
        else:
            self.gru_obs = GRUObservationCell(input_size, hidden_size, prep_hidden, bias=bias)

        self.covariates_map = torch.nn.Sequential(
            torch.nn.Linear(cov_size, cov_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(cov_hidden, hidden_size, bias=bias),
            torch.nn.Tanh()
        )

        assert solver in ["euler", "midpoint"], "vendored copy supports solver euler or midpoint"

        self.solver     = solver
        self.store_hist = options.pop("store_hist",False)
        self.input_size = input_size
        self.logvar     = logvar
        self.mixing     = mixing #mixing hyperparameter for loss_1 and loss_2 aggregation.

        self.apply(init_weights)

    def ode_step(self, h, p, delta_t, current_time):
        """Executes a single ODE step."""
        eval_times = torch.tensor([0],device = h.device, dtype = torch.float64)
        eval_ps = torch.tensor([0],device = h.device, dtype = torch.float32)
        if self.impute is False:
            p = torch.zeros_like(p)
            
        if self.solver == "euler":
            h = h + delta_t * self.gru_c(p, h)
            p = self.p_model(h)

        elif self.solver == "midpoint":
            k  = h + delta_t / 2 * self.gru_c(p, h)
            pk = self.p_model(k)

            h = h + delta_t * self.gru_c(pk, k)
            p = self.p_model(h)

        elif self.solver == "dopri5":
            assert self.impute==False #Dopri5 solver is only compatible with autonomous ODE.
            solution, eval_times, eval_vals = odeint(self.gru_c,h,torch.tensor([0,delta_t]),method=self.solver,options={"store_hist":self.store_hist})
            if self.store_hist:
                eval_ps = self.p_model(torch.stack([ev[0] for ev in eval_vals]))
            eval_times = torch.stack(eval_times) + current_time
            h = solution[1,:,:]
            p = self.p_model(h)
        
        current_time += delta_t
        return h,p,current_time, eval_times, eval_ps

        raise ValueError(f"Unknown solver '{self.solver}'.")

    def forward(self, times, time_ptr, X, M, obs_idx, delta_t, T, cov,
                return_path=False, smoother = False, class_criterion = None, labels=None):
        """
        Args:
            times      np vector of observation times
            time_ptr   start indices of data for a given time
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            obs_idx    observed patients of each datapoint (indexed within the current minibatch)
            delta_t    time step for Euler
            T          total time
            cov        static covariates for learning the first h0
            return_path   whether to return the path of h

        Returns:
            h          hidden state at final time (T)
            loss       loss of the Gaussian observations
        """

        h = self.covariates_map(cov)

        p            = self.p_model(h)
        current_time = 0.0
        counter      = 0

        loss_1 = 0 #Pre-jump loss
        loss_2 = 0 #Post-jump loss (KL between p_updated and the actual sample)

        if return_path:
            path_t = [0]
            path_p = [p]
            path_h = [h]

        if smoother:
            class_loss_vec = torch.zeros(cov.shape[0],device = h.device)
            num_evals_vec  = torch.zeros(cov.shape[0],device = h.device)
            class_criterion = class_criterion
            assert class_criterion is not None

        assert len(times) + 1 == len(time_ptr)
        assert (len(times) == 0) or (times[-1] <= T)

        eval_times_total = torch.tensor([],dtype = torch.float64, device = h.device)
        eval_vals_total  = torch.tensor([],dtype = torch.float32, device = h.device)

        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until next observation
            while current_time < (obs_time-0.001*delta_t): #0.0001 delta_t used for numerical consistency.
                 
                if self.solver == "dopri5":
                    h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, obs_time-current_time, current_time)
                else:
                    h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, delta_t, current_time)
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total  = torch.cat((eval_vals_total, eval_ps))

                #Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_p.append(p)
                    path_h.append(h)

            ## Reached an observation
            start = time_ptr[i]
            end   = time_ptr[i+1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = obs_idx[start:end]

            ## Using GRUObservationCell to update h. Also updating p and loss
            h, losses = self.gru_obs(h, p, X_obs, M_obs, i_obs)
           
            if smoother:
                class_loss_vec[i_obs] += class_criterion(self.classification_model(h[i_obs]),labels[i_obs]).squeeze(1)
                num_evals_vec[i_obs] +=1
            loss_1    = loss_1+ losses.sum()
            p         = self.p_model(h)

            loss_2 = loss_2 + compute_KL_loss(p_obs = p[i_obs], X_obs = X_obs, M_obs = M_obs, logvar=self.logvar)

            if return_path:
                path_t.append(obs_time)
                path_p.append(p)
                path_h.append(h)

        ## after every observation has been processed, propagating until T
        while current_time < T:
            if self.solver == "dopri5":
                h, p, current_time,eval_times, eval_ps = self.ode_step(h, p, T-current_time, current_time)
            else:
                h, p, current_time,eval_times, eval_ps = self.ode_step(h, p, delta_t, current_time)
            eval_times_total = torch.cat((eval_times_total,eval_times))
            eval_vals_total  = torch.cat((eval_vals_total, eval_ps))
            #counter += 1
            #current_time = counter * delta_t
            
            #Storing the predictions
            if return_path:
                path_t.append(current_time)
                path_p.append(p)
                path_h.append(h)

        loss = loss_1 + self.mixing * loss_2

        if smoother:
            class_loss_vec += class_criterion(self.classification_model(h),labels).squeeze(1)
            class_loss_vec /= num_evals_vec
        
        class_pred = self.classification_model(h)
       
        if return_path:
            if smoother:
                return h, loss, class_pred, np.array(path_t), torch.stack(path_p), torch.stack(path_h), class_loss_vec
            else:
                return h, loss, class_pred, np.array(path_t), torch.stack(path_p), torch.stack(path_h), eval_times_total, eval_vals_total
        else:
            if smoother:
                return h, loss, class_pred, class_loss_vec
            else:
                return h, loss, class_pred, loss_1

def compute_KL_loss(p_obs, X_obs, M_obs, obs_noise_std=1e-2, logvar=True):
    obs_noise_std = torch.tensor(obs_noise_std)
    if logvar:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        std = torch.exp(0.5*var)
    else:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        ## making var non-negative and also non-zero (by adding a small value)
        std       = torch.pow(torch.abs(var) + 1e-5,0.5)

    return (gaussian_KL(mu_1 = mean, mu_2 = X_obs, sigma_1 = std, sigma_2 = obs_noise_std)*M_obs).sum()


def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
    return(torch.log(sigma_2) - torch.log(sigma_1) + (torch.pow(sigma_1,2)+torch.pow((mu_1 - mu_2),2)) / (2*sigma_2**2) - 0.5)

