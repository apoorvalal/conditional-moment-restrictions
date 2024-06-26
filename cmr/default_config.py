import copy


"""Default configurations and hyperparameter search spaces for all methods"""


gel_kwargs = {
    "divergence": 'chi2',
    "reg_param": 0.0,
    "kernel_z_kwargs": {},
    "pretrain": False,
    "val_loss_func": 'moment_violation',

    # Optimization params
    "theta_optim_args": {"optimizer": "lbfgs", "lr": 5e-4},
    "dual_optim_args": {"optimizer": "lbfgs", "lr": 5 * 5e-4, "inneriters": 100},
    "max_num_epochs": 10000,
    "batch_size": None,
    "eval_freq": 200,
    "max_no_improve": 5,
    "burn_in_cycles": 10,
}

kmm_kwargs = {
    "divergence": 'kl',
    "entropy_reg_param": 1,
    "reg_param": 1.0,
    "kernel_x_kwargs": {},
    "n_random_features": 2000,
    "n_reference_samples": 200,
    "kde_bandwidth": 0.1,
    "annealing": False,
    "kernel_z_kwargs": {},
    "pretrain": True,
    "rkhs_func_z_dependent": True,
    "rkhs_reg_param": 1.0,
    "t_as_instrument": False,
    "val_loss_func": 'hsic',
    "gpu": False,

    # Optimization params
    "theta_optim_args": {"optimizer": "oadam_gda", "lr": 5e-4},
    "dual_optim_args": {"optimizer": "oadam_gda", "lr": 1e-4},
    "max_num_epochs": 10000,
    "batch_size": 200,
    "eval_freq": 100,
    "max_no_improve": 5,
    "burn_in_cycles": 10,
}

kmm_kernel_kwargs = copy.deepcopy(kmm_kwargs)
kmm_kernel_kwargs.update({"n_rff_instrument_func": 1000})

kmm_neural_kwargs = copy.deepcopy(kmm_kwargs)
kmm_neural_kwargs.update({"dual_func_network_kwargs": {}})

fgel_kernel_kwargs = {
    "divergence": 'chi2',
    "reg_param": 1e-6,
    "kernel_z_kwargs": {},
    "pretrain": True,
    "val_loss_func": 'hsic',

    # Optimization params
    "theta_optim_args": {"optimizer": "lbfgs", "lr": 5e-4},
    "dual_optim_args": {"optimizer": "lbfgs", "lr": 5 * 5e-4},
    "max_num_epochs": 10000,
    "batch_size": None,
    "eval_freq": 200,
    "max_no_improve": 5,
    "burn_in_cycles": 10,
}

fgel_neural_kwargs = {
    "divergence": 'chi2',
    "reg_param": 1.0,
    "dual_func_network_kwargs": {},
    "pretrain": True,
    "val_loss_func": 'hsic',

    # Optimization params
    "theta_optim_args": {"optimizer": "oadam_gda", "lr": 5e-4},
    "dual_optim_args": {"optimizer": "oadam_gda", "lr": 5 * 5e-4},
    "max_num_epochs": 10000,
    "batch_size": 200,
    "eval_freq": 100,
    "max_no_improve": 5,
    "burn_in_cycles": 10,
}

gmm_kwargs = {
    "reg_param": 1e-6,
    "num_iter": 2,
    "pretrain": True,
}

vmm_kernel_kwargs = {
    "reg_param": 1e-6,
    "num_iter": 2,
    "pretrain": True,
}

vmm_neural_kwargs = copy.deepcopy(fgel_neural_kwargs)
vmm_neural_kwargs.update({"reg_param_rkhs_norm": 0.0})


methods = {
    'OLS':
        {
            'estimator_kwargs': {},
            'hyperparams': {},
        },

    'GMM':
        {
            'estimator_kwargs': gmm_kwargs,
            'hyperparams': {'reg_param': [1e-8, 1e-6, 1e-4]}
        },

    'GEL':
        {
            'estimator_kwargs': gel_kwargs,
            'hyperparams': {"divergence": ['chi2', 'kl', 'log'],
                            "reg_param": [0, 1e-6]}
        },

    'MMR':
        {
            'estimator_kwargs': {"kernel_z_kwargs": {}},
            'hyperparams': {},
        },

    'SMD':
        {
            'estimator_kwargs': {},
            'hyperparams': {}
        },

    'DeepIV':
        {
            'estimator_kwargs': {},
            'hyperparams': {}
        },

    'VMM-kernel':
        {
            'estimator_kwargs': vmm_kernel_kwargs,
            'hyperparams': {'reg_param': [1e-8, 1e-6, 1e-4]}
        },

    'VMM-neural':
        {
            'estimator_kwargs': vmm_neural_kwargs,
            'hyperparams': {"reg_param": [0, 1e-4, 1e-2, 1e0],
            #"val_loss_func": ['mmr', 'moment_violation'],
                            }
        },

    'FGEL-kernel':
        {
            'estimator_kwargs': fgel_kernel_kwargs,
            'hyperparams': {
                'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
                "divergence": ['chi2', 'kl', 'log'],
            }
        },

    'FGEL-neural':
        {
            'estimator_kwargs': fgel_neural_kwargs,
            'hyperparams': {
                "reg_param": [0, 1e-4, 1e-2, 1e0],
                "divergence": ['chi2', 'kl', 'log'],
            }
        },

    'KMM':
        {
            'estimator_kwargs': kmm_kwargs,
            'hyperparams': {
                'entropy_reg_param': [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3],
            }
        },

    'KMM-kernel':
        {
            'estimator_kwargs': kmm_kernel_kwargs,
            'hyperparams': {
                'entropy_reg_param': [1e1, 1e0],
                'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
            }
        },

    'KMM-neural':
        {
            'estimator_kwargs': kmm_neural_kwargs,
            'hyperparams': {
                'entropy_reg_param': [1e0, 1e1, 1e2],
                "reg_param": [0, 1e-4, 1e-2, 1e0],
            }
        },
}

kmm_hyperparams = {"n_reference_samples": [200],    # [0, 100, 200, 400],
                   "entropy_reg_param": [1, 10],
                   "reg_param": [0.01, 0.1, 1],
                   # "kde_bandwidth": [0.1, 0.5, 1],  # [0.1, 1],
                   "n_random_features": [2000],    # [5000, 10000],
                   # #"val_loss_func": ['hsic'],
                   'theta_lr': [5e-4, 1e-4, 5e-5, 1e-5],
                   # 'dual_lr': [1e-4],
                   # 'batch_size': [200],
                   'max_num_epochs': [15000],
                   # "max_no_improve": [5],
                   }


def iterate_argument_combinations(argument_dict):
    args = list(argument_dict.values())
    pools = [tuple(pool) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield {key: [val] for key, val in zip(list(argument_dict.keys()), prod)}


# kmm_methods = {}
# for hparam in iterate_argument_combinations(kmm_hyperparams):
#     name = 'KMM'
#     for key, val in hparam.items():
#         name += f'_{key}_{val[0]}'
#     kmm_methods[name] = {'estimator_kwargs': kmm_neural_kwargs,
#                          'hyperparams': hparam, }

# FOR OPTIMIZATION TUNING ONLY
kmm_methods = {}
for config_id, hparam in enumerate(iterate_argument_combinations(kmm_hyperparams)):
    name = 'KMM'
    for key, val in hparam.items():
        name += f'_{key}_{val[0]}'
    estimator_kwargs = copy.deepcopy(kmm_neural_kwargs)
    estimator_kwargs.update({"theta_optim_args": {"optimizer": "oadam_gda", "lr": hparam['theta_lr'][0]},})
                            #"dual_optim_args": {"optimizer": "oadam_gda", "lr": hparam['dual_lr'][0]},})
    hparam['config'] = [config_id]
    kmm_methods[name] = {'estimator_kwargs': estimator_kwargs,
                         'hyperparams': hparam, }

methods.update(kmm_methods)

vmm_hyperparams = {"reg_param": [0, 1e-4, 1e-2, 1e0, 1e1],
                   "val_loss_func": ['mmr', 'moment_violation', 'hsic'], }

fgel_hyperparams = {
        "reg_param": [0, 1e-4, 1e-2, 1e0, 1e1],
        "divergence": ['chi2', 'kl', 'log'],
        "val_loss_func": ['mmr', 'moment_violation', 'hsic'],}

vmm_methods = {}
for hparam in iterate_argument_combinations(vmm_hyperparams):
    name = 'VMM-neural'
    for key, val in hparam.items():
        name += f'_{key}_{val[0]}'
    vmm_methods[name] = {'estimator_kwargs': vmm_neural_kwargs,
                         'hyperparams': hparam, }

methods.update(vmm_methods)

fgel_methods = {}
for hparam in iterate_argument_combinations(fgel_hyperparams):
    name = 'FGEL-neural'
    for key, val in hparam.items():
        name += f'_{key}_{val[0]}'
    fgel_methods[name] = {'estimator_kwargs': fgel_neural_kwargs,
                         'hyperparams': hparam, }

methods.update(fgel_methods)


if __name__ == '__main__':
    print(kmm_methods)
