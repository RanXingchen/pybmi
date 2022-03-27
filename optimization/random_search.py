import random
import numpy as np
import torch
import json


class RandomSearch():
    def __init__(self, params_grid: dict, max_iters: int = 500, bsmall=True):
        """
        Random search for the optimal hyper-parameters.

        Parameters
        ----------
        params_grid : dict
            The parameter grid that contain all desired combinations of
            hyperparameters.
        max_iters : int, optional
            The maximum iterations for random search.

        Attributes
        ----------
        best_score : float
            The best score for the searching result.
        best_hyper_params : dict
            The optimal combination of the searching result,
            which detemined by the BEST_SCORE.
        best_model : any
            The optimal model of the searching result, which detemined by the
            BEST_SCORE. The model type is determined by the user specified
            training function.
        best_iter : int
            The number of iteration of best scores.
        """
        self.params_grid = params_grid
        self.max_iters = max_iters
        self.bsmall = bsmall

        # Recording the best parameters
        self.best_score = np.Inf if bsmall else -np.Inf
        self.best_hyper_params = {}
        self.best_model = None
        self.best_iter = -1

    def __call__(self, train_fn, save_name=None, verbose=True):
        """
        Parameters
        ----------
        train_fn : function
            User's training function, which takes the dictionary of all
            hyperparameters as input. And the output of this function should
            be the MODEL and the validation SCORE.
        save_name : str
            The save name of the BEST_MODEL.
        verbose : bool
            Whether to show the progress in screen.

        Returns
        -------
        This function returns the BEST_MODEL and BEST_SCORE.
        """
        for i in range(self.max_iters):
            # The selection of the parameter conbination is determined
            # by the the number i.
            random.seed(i)
            # Obtain the random parameters.
            hyper_params = {k: random.sample(v, 1)[0]
                            for k, v in self.params_grid.items()}

            if verbose:
                print(f'\n=====Iteration {i + 1}=====')
                print('Current searching hyperparameters combination is:')
                print(json.dumps(hyper_params, indent=4))
            # Run the training function, this function only take the
            # parameters as input. And it returns the models and cv scores.
            model, score = train_fn(hyper_params)
            # Update the best results.
            res = self.best_score > score if self.bsmall else \
                self.best_score < score
            if res:
                self.best_hyper_params = hyper_params
                self.best_score = score
                self.best_model = model
                self.best_iter = i + 1
                # Save the best models
                if save_name is not None:
                    torch.save(model.state_dict(), save_name)
                # Show the best parameters in screen.
                if verbose:
                    print(f'End of {i + 1} iteration.'
                          f'Current best cv score: {self.best_score}.')
                    print('Current best hyperparameters combination'
                          f'is from {self.best_iter}: ')
                    print(json.dumps(self.best_hyper_params, indent=4))
        return self.best_model, self.best_score
