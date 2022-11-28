'''
Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
import gin
from ray.tune.sample import loguniform, uniform, randint, Categorical

import ConfigSpace as CS


@gin.configurable
def paramSearch(space={}, default_config={}, numTrials=None, space_name="ConfigSpace"):
    default_config.update(space)
    if space_name == "ConfigSpace":
        search_space = CS.ConfigurationSpace()
    else:
        search_space = {}
    for k, values in default_config.items():
        if space_name == "ConfigSpace":
            if isinstance(values, tuple):
                (space_type, lower, upper) = values
                if space_type == "float":
                    search_space.add_hyperparameter(
                        CS.UniformFloatHyperparameter(k, lower=lower, upper=upper)
                    )
                elif space_type == "logfloat":
                    search_space.add_hyperparameter(
                        CS.UniformFloatHyperparameter(
                            k, lower=lower, upper=upper, log=True
                        )
                    )
                elif space_type == "int":
                    search_space.add_hyperparameter(
                        CS.UniformIntegerHyperparameter(k, lower=lower, upper=upper)
                    )
                else:
                    raise NotImplementedError
            else:
                search_space.add_hyperparameter(CS.Constant(k, values))
        elif space_name == "Ray":
            if isinstance(values, tuple):
                (space_type, lower, upper) = values
                if space_type == "float":
                    search_space[k] = uniform(lower, upper)
                elif space_type == "logfloat":
                    search_space[k] = loguniform(lower, upper)
                elif space_type == "int":
                    search_space[k] = randint(lower, upper)
                else:
                    raise NotImplementedError
            else:
                search_space[k] = Categorical([values]).uniform()
        elif space_name == "Sampler":
            if isinstance(values, tuple):
                (space_type, lower, upper) = values
                if space_type == "float":
                    search_space[k] = uniform(lower, upper).sample
                elif space_type == "logfloat":
                    search_space[k] = loguniform(lower, upper).sample
                elif space_type == "int":
                    search_space[k] = randint(lower, upper).sample
                else:
                    raise NotImplementedError
            else:
                search_space[k] = Categorical([values]).uniform().sample
    return list(space.keys()), search_space, numTrials
