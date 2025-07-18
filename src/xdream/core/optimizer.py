'''
This file contains the implementation of the `Optimizer` class and its subclasses.
It provides a set of classes that implement different optimization strategies:
- `GeneticOptimizer`: Optimizer that implements a genetic optimization strategy.
- `CMAESOptimizer`: Optimizer that implements a Covariance Matrix Adaptation Evolution Strategy.
'''

from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax
from cma import CMAEvolutionStrategy

from xdream.core.utils.types import Codes, Fitness
from xdream.core.utils.misc import default

RandomDistribution = Literal['normal', 'gumbel', 'laplace', 'logistic']
''' 
Name of distributions for random codes initializations
'''

class Optimizer(ABC):
    '''
    Abstract class for a generic optimizer intended to maximize an objective.
    It implements a function `step()` to produce a new set of codes based on a scoring input.
    '''
    
    '''
    TODO Add back for future implementation
    
    `states_space`: None | Dict[int | str, Tuple[float | None, float | None]] = None
    Dictionary specifying the optimization domain where 
    each entry is a direction and corresponding values
    are the min-max acceptable values along that dir, defaults to None.
    
    `states_shape`: None | int | Tuple[int, ...] = None
    Dictionary specifying the optimization domain where 
    each entry is a direction and corresponding values are 
    the min-max acceptable values along that direction, defaults to None.
    
    if not (states_shape or states_space):
            err_msg = 'Either `states_shape` or `states_space` must be specified, but both weren\'t.'
            raise ValueError(err_msg)
                
        # States shape - int to tuple conversion
        if isinstance(states_shape, int):
            states_shape = (states_shape,)
            
    self._space = lazydefault(states_space, lambda : {i : (None, None) for i in range(len(states_shape))})  # type: ignore
    self._shape = lazydefault(states_shape, lambda : (len(states_space),))                                  # type: ignore
    '''
    
    # --- INIT ---
    
    def __init__(
        self,
        pop_size    : int,
        codes_shape : int | Tuple[int, ...],
        rnd_seed    : None | int = None,
        rnd_distr   : RandomDistribution = 'normal',
        rnd_scale   : float = 1.
    ) -> None:
        '''
        Initialize a new gradient free optimizer with proper population size and codes shape.

        :param pop_size: Number of initial codes.
        :type pop_size: int
        :param codes_shape: Codes shape. If one dimensional the single dimension supported.
        :type codes_shape: int | Tuple[int, ...]
        :param rnd_seed: Random state for pseudo-random numbers generation.
        :type rnd_seed: None | int, optional
        :param rnd_distr: Nature of the random distribution for initial
            random codes generation, defaults to `normal`.
        :type rnd_distr: RandomDistribution
        :param rnd_scale: Scale for initial codes generation, defaults to 1.
        :type rnd_scale: float
        '''
        
        # Save initial number of codes
        self._init_n_codes = pop_size
        
        # Codes shape with single dimension cast
        if isinstance(codes_shape, int):
            codes_shape = codes_shape,
        self._codes_shape = codes_shape
        
        # Randomic components
        self._rnd_seed   = rnd_seed
        self._rng        = np.random.default_rng(rnd_seed)
        self._rnd_scale  = rnd_scale
        self._rnd_sample = self._get_rnd_sample(distr  = rnd_distr)
        
        # Last generated codes
        self._codes : None | Codes = None
        
    # --- PROPERTIES ---
    
    @property
    def codes_len(self) -> int: return int(np.prod(self._codes_shape))
        
    @property
    def codes(self) -> Codes:
        '''
        Returns codes produced in the last step.
        It raises an error in the case no codes are available.
        
        Codes are internally linearized, the property
        handles codes reshaping to the expected to the expected shape.
        
        :return: Last produced codes.
        :rtype: Codes
        '''
        
        # Codes not available check
        if self._codes is None:
            err_msg = 'No codes available. Use `init()` method to generate the first codes.'
            raise ValueError(err_msg)
        
        # Extract population size
        pop_size, *_ = self._codes.shape
        
        # Reshape codes to the expected shape
        codes_ = np.reshape(self._codes, (pop_size, *self._codes_shape))

        return codes_.copy()
    
    
    @property
    def pop_size(self) -> int:
        ''' 
        Number of codes the optimizer is optimizing for.
        NOTE:   The number of codes can change dynamically 
                during the optimization process
        '''
        
        try:
            pop_size, *_ = self.codes.shape
            return pop_size
        
        except ValueError:
            return self._init_n_codes
    
    # --- STEP ---
    
    def step(self, scores: Fitness) -> Codes:
        '''
        Wrapper for actual step implementation in `_step()` that automatizes 
        saving of last generation codes in `self_codes`.
        
        :param scores: Tuple containing the score associated to each old code.
        :type scores: Score
        :return: Set of new codes supposed to produce an higher value of the objective.
        :rtype: Codes
        ''' 
        
        self._codes = self._step(scores=scores)
        
        return self.codes
    
    
    @abstractmethod
    def _step(self, scores: Fitness) -> Codes:
        '''
        Abstract step method.
        The `step()` method receives the scores associated to the
        last produced codes and uses them to produce a new set of codes.
        
        By defaults it only checks if codes are available.
        
        :param scores: Tuple containing the score associated to each old code.
        :type scores: Scores
        :return: Set of new codes to be used to improve future states scores.
        :rtype: Codes
        '''        
        
        if self._codes is None:
            err_msg = 'No codes provided, use `init()` to generate first ones. '
            raise ValueError(err_msg)
        
        pass
    
    
    # --- CODE INITIALIZATION ---
    
    def init(
        self, 
        init_codes : NDArray | None = None, 
        **kwargs
    ) -> Codes:
        '''
        Initialize the optimizer codes. 
        
        If initial codes are provided as arrays they should have matching
        dimensionality as expected by the provided states shape,
        otherwise they are randomly sampled.
        
        :param init_codes: Initial codes for optimizations, optional.
        :type init_codes: NDArray | None.
        :param kwargs: Parameters that are passed to the random
            generator to sample from the chosen distribution
            (e.g. loc=0, init_cond=normal).
        '''
        
        # Codes provided
        if isinstance(init_codes, np.ndarray):
            
            # Check shape consistency
            exp_shape = (self.pop_size, *self._codes_shape)
            if init_codes.shape != exp_shape:
                err_msg =   f'Provided initial codes have shape: {init_codes.shape}, '\
                            f'do not match expected shape {exp_shape}'
                raise Exception(err_msg)
            
            # Use input codes as first codes
            self._codes = init_codes
            
        # Codes not provided: random generation
        else:
            
            # Generate codes using specified random distribution
            self._codes = self._rnd_codes_generation(**kwargs)

        return self.codes
    
    def _rnd_codes_generation(self, **kwargs):
        '''
        Generate random codes using the specified distribution.
        It uses additional parameters passed as kwargs to the random generator.
        
        :return: Randomly generated codes.
        :rtype: Codes
        '''
        
        return self._rnd_sample(
            size=(self._init_n_codes, self.codes_len),
            scale=self._rnd_scale,
            **kwargs
        )
        
    
    def _get_rnd_sample(
        self,
        distr : RandomDistribution = 'normal'
    ) -> Callable:
        '''
        Uses the distribution input attributes to return 
        the specific distribution function.
        
        :param distr: Random distribution type.
        :type distr: RandomDistribution
        :param scale: Random distribution scale.
        :type scale: float.
        :return: Distribution function.
        :rtype: Callable
        '''
        
        match distr:
            case 'normal':   return self._rng.normal
            case 'gumbel':   return self._rng.gumbel
            case 'laplace':  return self._rng.laplace
            case 'logistic': return self._rng.logistic
            case _: raise ValueError(f'Unrecognized distribution: {distr}')
    
class GeneticOptimizer(Optimizer):
    '''
    Optimizer that implements a genetic optimization strategy.
    
    In particular these optimizer devise a population of candidate
    solutions (set of parameters) and iteratively improves the
    given objective function via the following heuristics:
    
    - The top_k performing solution are left unaltered
    - The rest of the population pool are recombined to produce novel
        candidate solutions via breeding and random mutations
    - The n_parents contributing to a single offspring are selected
        via importance sampling based on parents fitness scores
    - Mutations rate and sizes can be adjusted independently 
    '''
    
    def __init__(
        self,
        codes_shape  : int | Tuple[int, ...],
        rnd_seed     : None | int = None,
        rnd_distr    : RandomDistribution = 'normal',
        rnd_scale    : float = 1.,
        pop_size     : int   = 50,
        mut_size     : float = 0.1,
        mut_rate     : float = 0.3,
        n_parents    : int   = 2,
        allow_clones : bool  = False,
        topk         : int   = 2,
        temp         : float = 1.,
        temp_factor  : float = 1.,
    ) -> None:
        '''
        Initialize a new GeneticOptimizer
        
        :param pop_size: Number of initial codes.
        :type pop_size: int
        :param codes_shape: Codes shape. If one dimensional providing the single dimension is supported.
        :type codes_shape: int | Tuple[int, ...]
        :param rnd_seed: Random state for pseudo-random numbers generation.
        :type rnd_seed: None | int, optional
        :param rnd_distr: Nature of the random distribution for initial
            random codes generation, defaults to `normal`.
        :type rnd_distr: RandomDistribution
        :param rnd_scale: Scale for initial codes generation, defaults to 1.
        :type rnd_scale: float
        :param pop_size: Number of codes in the population, defaults to 50
        :type pop_size: int, optional
        :param mut_size: Probability of single-point mutation, defaults to 0.3
        :type mut_size: float, optional
        :param mut_rate: Scale of punctual mutations (how big the effect of 
            mutation can be), defaults to 0.1
        :type mut_rate: float, optional
        :param n_parents: Number of parents contributing their genome
            to a new individual, defaults to 2
        :type n_parents: int, optional
        :param allow_clones: If a code can occur as a parent multiple times when more 
            than two parents are used, default to False.
        :type allow_clones: bool, optional
        :param temp: Temperature for controlling the softmax conversion
            from scores to fitness (the actual prob. to sample 
            a given parent for breeding), defaults to 1.
        :type temp: float, optional
        :param temp_factor: Multiplicative factor for temperature increase (`temp_factor` > 1)  
            or decrease (0 < `temp_factor` < 1). Defaults to 1. indicating no change.
        :type temp: float, optional
        '''
        
        # TODO Parameter domain sanity check
        
        super().__init__(
            pop_size=pop_size,
            codes_shape=codes_shape,
            rnd_seed=rnd_seed,
            rnd_distr=rnd_distr,
            rnd_scale=rnd_scale
        )
        
        # Optimization hyperparameters
        self._mut_size     = mut_size
        self._mut_rate     = mut_rate
        self._n_parents    = n_parents
        self._allow_clones = allow_clones
        self._topk         = topk
        self._temp         = temp
        self._temp_factor  = temp_factor
        
    # --- STRING REPRESENTATION ---
    
    def __str__(self) -> str:
        ''' Return a string representation of the object for logging'''
        
        return  f'GeneticOptimizer['\
                f'mut_size: {self._mut_size}'\
                f'mut_rate: {self._mut_rate}'\
                f'n_parents: {self._n_parents}'\
                f'allow_clones: {self._allow_clones}'\
                f'topk: {self._topk}'\
                f'temp: {self._temp}'\
                f'temp_factor: {self._temp_factor}'\
                ']'
    
    def __repr__(self) -> str: return str(self)
    ''' Return a string representation of the object''' 
    
    def _step(
        self,
        scores : Fitness,
        out_pop_size : int | None = None  
    ) -> Codes:
        '''
        Optimizer step function that uses an associated score
        to each code to produce a new set of stimuli.

        :param scores: Scores associated to each code.
        :type scores: Score
        :param out_pop_size: Population size for the next generation. 
            Defaults to old one.
        :type out_pop_size: int | None, optional
        :return: Optimized set of codes.
        :rtype: Score
        '''
        
        super()._step(scores=scores)
        
        # Use old population size as default
        pop_size = default(out_pop_size, self.pop_size)      

        # Prepare data structure for the optimized codes
        codes_new = np.empty(shape=(pop_size, self.codes_len), dtype=np.float32)

        # Get indices that would sort scores so that we can use it
        # to preserve the top-scoring stimuli
        sort_s = np.argsort(scores)
        topk_old_gen = self._codes[sort_s[-self._topk:]]  # type: ignore - check made in super function
        
        
       
        # Convert scores to fitness (probability) via 
        # temperature-gated softmax function (needed only for rest of population)
        fitness = softmax(scores / self._temp)
        
        # The rest of the population is obtained by generating
        # children using breeding and mutation.
        
        # Breeding
        new_gen = self._breed(
            population=self._codes.copy(),  # type: ignore
            pop_fitness=fitness,
            num_children=pop_size-self._topk,
        )
        
        # Mutating
        new_gen = self._mutate(
            population=new_gen
        )

        # New codes combining previous top-k codes and new generated ones
        codes_new[:self._topk] = topk_old_gen
        codes_new[self._topk:] = new_gen
        
        # Temperature 
        self._temp *= self._temp_factor
        
        # TODO: Add temperature rewarming of a factor `temp_rewarm()` if below treshold `temp_threshold`
        
        self._codes = codes_new.copy()
        
        return codes_new

    def _breed(
        self,
        population  : NDArray,
        pop_fitness : NDArray,
        num_children : int | None = None
    ) -> NDArray:
        '''
        Perform breeding on the given population with given parameters.

        :param population: Population to breed using the fitness.
        :type population: NDArray.
        :param pop_fitness: Population fitness (i.e. probability to be selected
            as parents for the next generation).
        :type pop_fitness: NDArray.
        :param num_children: Number of children in the new population, defaults to the
            total number of codes (no parent preserved)
        :type num_children: int | None, optional
        
        :return: Breed population.
        :rtype: NDArray
        '''
        
        # NOTE: Overwrite old population

        # Number of children defaults to population size
        # i.e. no elements in the previous generation survives in the new one
        num_children = default(num_children, self.pop_size)
        
        # We use clones if specified and if the number of parents is greater than 2
        use_clones = self._allow_clones and self._n_parents > 2
        
        families = self._rng.choice(
            a=self.pop_size,
            size=(num_children, self._n_parents),
            p=pop_fitness,
            replace=True 
        # Otherwise we sample one element at a time without replacement
        # and combine them to form the family
        ) if use_clones else np.stack([
            self._rng.choice(
                a = self.pop_size,
                size=self._n_parents,
                p=pop_fitness,
                replace=False,
            ) for _ in range(num_children)
        ])

        # Identify which parent contributes which genes for every child
        parentage = self._rng.choice(
            a=self._n_parents, 
            size=(num_children, self.codes_len), 
            replace=True
        )
        
        # Generate empty children
        children = np.empty(shape=(num_children, self.codes_len))

        # Fill children with genes from selected parents
        for child, family, lineage in zip(children, families, parentage):
            for i, parent in enumerate(family):
                genes = lineage == i
                child[genes] = population[parent][genes]
                
        return children

    def _mutate(
        self,
        population : NDArray
    ) -> NDArray:
        '''
        Perform punctual mutation to given population using input parameters.

        :param population: Population of codes to mutate.
        :type population: NDArray
        :return: Mutated population.
        :rtype: NDArray
        '''

        # Compute mutation mask
        mut_loc = self._rng.choice(
            [True, False],
            size=population.shape,
            p=(self._mut_rate, 1 - self._mut_rate),
            replace=True
        )

        # Apply mutation
        population[mut_loc] += self._rnd_sample(
            scale=self._mut_size, 
            size=mut_loc.sum()
        )
        
        return population 



class CMAESOptimizer(Optimizer):
    
    def __init__(
        self,
        pop_size    : int,
        codes_shape: int | Tuple[int, ...],
        rnd_seed   : None | int = None,
        x0         : NDArray | None = None,
        sigma0     : float = 1.,
    ) -> None:
        '''
        Initialize a new CMAESOptimizer with initial Multivariate gaussian
        mean vector and variance for the covariance matrix as initial parameters.

        :param pop_size: Number of initial codes.
        :type pop_size: int
        :param codes_shape: Codes shape. If one dimensional the single dimension supported.
        :type codes_shape: int | Tuple[int, ...]
        :param rnd_seed: Random state for pseudo-random numbers generation.
        :type rnd_seed: None | int, optional
        :param x0: Initial mean vector for the multivariate gaussian distribution, defaults to None that 
            is a zero mean vector.
        :type x0: NDArray | None, optional
        :param sigma0: Initial variance for the covariance matrix, defaults to 1.
        :type sigma0: float, optional
        '''
        
        super().__init__(
            pop_size=pop_size, 
            codes_shape=codes_shape,
            rnd_seed=rnd_seed
        )
        
        # Save variance for the covariance matrix
        self._sigma0 = sigma0
        
        # Use zero mean vector if not provided
        x0 = default(x0, np.zeros(shape=self.codes_len))
        
        # Create dictionary for CMA-ES settings
        inopts = {'popsize': pop_size}
        if rnd_seed: inopts['seed'] = rnd_seed
        
        # Initialize CMA-ES optimizer
        self._es = CMAEvolutionStrategy(
            x0     = x0,
            sigma0 = sigma0,
            inopts = inopts
        )
    
    # --- STRING REPRESENTATION ---
    def __str__ (self) -> str: return f'CMAESOptimizer[sigma0: {self._sigma0}]'
    ''' Return a string representation of the object '''
    
    def __repr__(self) -> str: return str(self)
    ''' Return a string representation of the object '''


    # --- STEP ---

    def _step(self, scores: Fitness) -> Codes:
        '''
        Perform a step of the optimization process using the CMA-ES optimizer.

        :param scores: Tuple containing the score associated to each old code.
        :type scores: Scores
        :return: New set of codes.
        :rtype: Codes
        '''
        
        super()._step(scores=scores)
        
        self._es.tell(
            solutions=list(self._codes.copy()), # type: ignore
            function_values=list(-scores)
        )
        
        self._codes = np.stack(self._es.ask())
        
        return self._codes
    
    def _rnd_codes_generation(self, **kwargs) -> Codes:
        '''
        Override super method to generate random codes using the CMA-ES optimizer.

        :return: Randomly generated codes using current CMA-ES optimizer state.
        :rtype: Codes
        '''
        
        return np.stack(self._es.ask())
    
### GIACKO'S OPTIMIZERS <3 ###

######## PSO ########

from sklearn.cluster import KMeans

class PSOOptimizer(Optimizer):
    """
    Optimizer that implements Particle Swarm Optimization (PSO),
    now with optional cluster-based partial updates.
    """

    def __init__(
        self,
        codes_shape : int | Tuple[int, ...],
        pop_size    : int = 50,
        inertia_max : float = 0.99,
        inertia_min : float = 0.75,
        cognitive   : float = 1.2,
        social      : float = 2.0,
        v_clip      : float = 0.15,
        num_informants : int = 15,
        first_PSO_interval: float = 0.5,
        second_PSO_interval: float = 0.85,
        bounds      : None | NDArray = None,
        rnd_seed    : None | int = None,
        rnd_distr   : RandomDistribution = 'normal',
        rnd_scale   : float = 1.,
        max_iterations: int = 500,
        
        # New parameters for clustering
        use_clustering : bool = False,
        n_clusters     : int  = 5,
        top_k          : int  = 3,
    ) -> None:
        super().__init__(
            pop_size=pop_size,
            codes_shape=codes_shape,
            rnd_seed=rnd_seed,
            rnd_distr=rnd_distr,
            rnd_scale=rnd_scale,
        )
        self.inertia_max = inertia_max
        self.inertia_min = inertia_min
        self.inertia = inertia_max
        self.cognitive = cognitive
        self.social = social
        self.v_clip = v_clip
        self.num_informants = num_informants
        self.bounds = bounds
        self.velocity = np.zeros((pop_size, self.codes_len))
        self.best_positions = np.zeros((pop_size, self.codes_len))
        self.best_scores = np.full(pop_size, -np.inf)
        self.global_best_position = np.zeros(self.codes_len)
        self.global_best_score = -np.inf
        self.informants = self._initialize_informants()
        self._rng = np.random.default_rng(rnd_seed)
        self.iteration = 0
        self.max_iterations = max_iterations
        self.first_PSO_interval = first_PSO_interval
        self.second_PSO_interval = second_PSO_interval
        self.history = {
            'mean_fitness': [],
            'best_fitness': [],
        }
        
        # Clustering config
        self.use_clustering = use_clustering
        self.n_clusters = n_clusters
        self.top_k = top_k

    def _initialize_informants(self):
        """
        Initialize the informants for each particle (i.e. the social
        behavior of the solution).
        """
        informants = []
        for i in range(self.pop_size):
            other_indices = [j for j in range(self.pop_size) if j != i]
            selected_informants = self._rng.choice(
                other_indices,
                size=min(self.num_informants - 1, self.pop_size -1),
                replace=False
            )
            informant_indices = np.concatenate(([i], selected_informants))
            informants.append(informant_indices)
        return informants

    # NON-LINEAR INERTIA SCHEDULER
    def _update_inertia(self):
        # Only update after we have at least 2 data points
        #if len(self.history['mean_fitness']) > 1:
        #    mean_prev = self.history['mean_fitness'][-2]
        #    mean_cur = self.history['mean_fitness'][-1]
        #    
        #    threshold = 0.05*abs(mean_prev)  # e.g. 10% of previous mean  0.1 *
        #    delta_fitness = abs(mean_cur - mean_prev)
        #    
        #    if delta_fitness < threshold:
        #        # Grow inertia, but more gently
        #        self.inertia = min(self.inertia_max, self.inertia + 0.01)
        #    else:
                # Shrink inertia, but not too drastically
        if self.iteration % 10:    
            self.inertia = max(self.inertia_min, self.inertia - 0.005)
            
        
        self.iteration += 1

    def _cluster_and_get_topk_indices(self, scores: Fitness) -> np.ndarray:
        """
        1) Cluster the population via KMeans.
        2) Within each cluster, pick indices of the top-k by `scores`.
        3) Return the combined list of those indices (unique).
        
        If top_k * n_clusters > pop_size, some clusters might have fewer 
        than top_k if cluster sizes are small. 
        """
        # Edge case: if n_clusters >= pop_size, skip clustering
        if self.n_clusters >= self.pop_size:
            # Fall back to entire population or some simpler strategy
            return np.arange(self.pop_size)

        # Cluster the codes
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self._rng.integers(1e9), n_init=10)
        labels = kmeans.fit_predict(self._codes)  # shape = (pop_size,)

        # For each cluster, find top-k
        selected_indices = []
        for cluster_label in range(self.n_clusters):
            cluster_indices = np.where(labels == cluster_label)[0]
            if len(cluster_indices) == 0:
                continue  # no members in this cluster

            # Sort by descending score
            sorted_cluster_indices = cluster_indices[np.argsort(scores[cluster_indices])[::-1]]
            # Pick top_k
            best_in_cluster = sorted_cluster_indices[:self.top_k]
            selected_indices.extend(best_in_cluster.tolist())
           

        return np.array(selected_indices)

    def _partial_pso_update(self, indices: np.ndarray) -> None:
        """
        Perform the velocity and position update only for
        particles at the specified `indices`.
        """
        for i in indices:
            # Standard PSO velocity updates
            cognitive_rand = self._rng.uniform(size=self.codes_len)
            cognitive_component = self.cognitive * (self.best_positions[i] - self._codes[i])
            
            informant_best_idx = np.argmax(self.best_scores[self.informants[i]])
            informant_best_position = self.best_positions[self.informants[i][informant_best_idx]]
            social_rand = self._rng.uniform(size=self.codes_len)
            social_component = self.social * (informant_best_position - self._codes[i])

            self.velocity[i] = (
                self.inertia * self.velocity[i]
                + cognitive_component
                + social_component
            )
            self.velocity[i] = np.clip(self.velocity[i], -self.v_clip, self.v_clip)
            self._codes[i] += self.velocity[i]
           

    def _step(self, scores: Fitness, update_inertia: bool = True) -> Codes:
        if update_inertia:
            self._update_inertia()

        # Update personal and global bests for all
        for i in range(self.pop_size):
            if scores[i] > self.best_scores[i]:
                self.best_scores[i] = scores[i]
                self.best_positions[i] = self._codes[i].copy()

            if scores[i] > self.global_best_score:
                self.global_best_score = scores[i]
                self.global_best_position = self._codes[i].copy()

        # If we are NOT using cluster-based updates, 
        # do the standard PSO on the entire population:
        if not self.use_clustering:
            all_indices = np.arange(self.pop_size)
            self._partial_pso_update(all_indices)

        else:
            # Cluster and select top_k from each cluster
            topk_indices = self._cluster_and_get_topk_indices(scores)
            # Only update these top-k in each cluster
            self._partial_pso_update(topk_indices)

        # Append metrics
        self.history['mean_fitness'].append(np.mean(scores))
        self.history['best_fitness'].append(np.max(scores))

        return self._codes




######## Hybrid ########

class HybridOptimizer(Optimizer):

    '''

    Optimizer that implements a hybrid optimization strategy combining 

    Particle Swarm Optimization (PSO) and Genetic Algorithm (GA).

    '''

    def __init__(

        self,

        codes_shape     : int | Tuple[int, ...],

        pop_size        : int = 50,

        ga_params       : dict | None = None,

        pso_params      : dict | None = None,

        rnd_seed        : None | int = None,

        rnd_distr       : RandomDistribution = 'normal',

        rnd_scale       : float = 1.,

        max_iterations  : int = 100,

      

        diversity_threshold: float = 0.01,  #   Minimum diversity threshold

        stagnation_threshold: float = 0.1, #   Fitness improvement threshold

        first_PSO_interval: float = 0.3,   #   First threshold for PSO frequency

        second_PSO_interval: float = 0.8,  #   Second threshold for PSO frequency

        less_cog_factor: float = 0.995,     #   Factor to decrease cognitive component

        more_cog_factor: float = 1.05,     #   Factor to increase cognitive component

        more_mut_rate: float = 1.005,        #   Factor to increase mutation rate

        less_mut_rate: float = 0.995,        #   Factor to decrease mutation rate

        less_mut_size: float = 0.955,       #   Factor to decrease mutation size

    ) -> None:

        super().__init__(

            pop_size=pop_size,

            codes_shape=codes_shape,

            rnd_seed=rnd_seed,

            rnd_distr=rnd_distr,

            rnd_scale=rnd_scale,

        )

        if ga_params is None:

            ga_params = {

                'mut_size': 0.6,
                'mut_rate': 0.6,

                'n_parents': 4,

                'allow_clones': True,   # True

                'topk': 5,              # 2

                'temp': 1.2,

                'temp_factor': 0.98,

            }

        if pso_params is None:

            pso_params = {

                #  PSO parameters

                'inertia_max': 0.99,        # Maximum inertia weight

                'inertia_min': 0.95,        # Minimum inertia weight

                'cognitive': 2.5,          # Cognitive component weight

                'social': 2.0,             # Social component weight

                'v_clip': 0.15,            # Velocity clipping threshold

                'num_informants': 20,       # Number of informant particles

                'bounds': None,            # Search space bounds

                'max_iterations': 500,  # Maximum number of iterations

                'first_PSO_interval': 0.6, # First threshold for PSO frequency

                'second_PSO_interval': 0.85,

                

                # ga-specific parameters 

                'ga_params': {

                    'mut_size': 0.6,      # Mutation size for genetic component

                    'mut_rate': 0.6,      # Mutation rate

                    'n_parents': 4,       # Number of parents for crossover

                    'allow_clones': True, # Allow duplicate solutions

                    'topk': 2,          # Top k selection parameter

                    'temp': 1.2,         # Temperature for selection

                    'temp_factor': 0.98,  # Temperature decay factor

                },

                

                # adaptative parameters

                

                'diversity_threshold': 0.01, # Minimum diversity threshold

                'stagnation_threshold': 0.01, # Fitness improvement threshold

}











        self.hyperparams = {

    

            'inertia_max': 0.99,        # Maximum inertia weight

            'inertia_min': 0.95,        # Minimum inertia weight

            'cognitive': 2.5,          # Cognitive component weight

            'social': 2.0,             # Social component weight

            'v_clip': 0.15,            # Velocity clipping threshold

            'num_informants': 20,       # Number of informant particles

            

            

            'mut_size': 0.6,      # Mutation size for genetic component

            'mut_rate': 0.6,      # Mutation rate

            'n_parents': 4,       # Number of parents for crossover

            

            'topk': 5,          # Top k selection parameter, 2

            'temp': 1.2,         # Temperature for selection

            'temp_factor': 0.98,

            

            

            'diversity_threshold': diversity_threshold,

            'stagnation_threshold': stagnation_threshold,

        }    



        self.ga_params = ga_params

        self.pso_params = pso_params

        self._rng = np.random.default_rng(rnd_seed)


        self.first_PSO_interval = first_PSO_interval

        self.second_PSO_interval = second_PSO_interval

        self.less_cog_factor = less_cog_factor

        self.more_cog_factor = more_cog_factor

        self.more_mut_rate = more_mut_rate

        self.less_mut_rate = less_mut_rate

        self.less_mut_size = less_mut_size

        

        self.ga_optimizer = GeneticOptimizer(

            codes_shape=codes_shape,

            pop_size=pop_size,

            rnd_seed=rnd_seed,

            rnd_distr=rnd_distr,

            rnd_scale=rnd_scale,

            **ga_params

        )

        

        self.pso_optimizer = PSOOptimizer(

            codes_shape=codes_shape,

            pop_size=pop_size,

            rnd_seed=rnd_seed,

            rnd_distr=rnd_distr,

            rnd_scale=rnd_scale,
            use_clustering=True,
            n_clusters=5,
            top_k=2,

            **pso_params

        )

        self._codes = self.ga_optimizer.init()      # Initialize codes with GA   

        self.pso_optimizer._codes = self._codes.copy()

        self.max_iterations = max_iterations

        self.current_iteration = 0

        self.diversity_threshold = diversity_threshold

        self.stagnation_threshold = stagnation_threshold

        

        

        self.history = {

            'mean_fitness': [],

            'best_fitness': [],

     

        }

                

        

    



    def _update_pso_after_ga(self, scores: Fitness):

        '''

        Update the PSO optimizer after the GA step.

        '''

        for i in range(self.pop_size):

            # Update best scores and best positions if the current GA score is better

            if scores[i] > self.pso_optimizer.best_scores[i]:

                self.pso_optimizer.best_scores[i] = scores[i]

                self.pso_optimizer.best_positions[i] = self._codes[i].copy()

        # Update global best

        max_scores_idx = np.argmax(scores)

        

        if scores[max_scores_idx] > self.pso_optimizer.global_best_score:

           

            self.pso_optimizer.global_best_score = scores [max_scores_idx]

            self.pso_optimizer.global_best_position = self._codes[max_scores_idx].copy()

            



    

    def _reset_velocity_after_ga(self):

        """

        Reset PSO velocities after GA step based on recent successful directions.

        this is 

        """

        # Calculate the difference between current positions (After GA) and personal bests

        position_deltas = self.pso_optimizer.best_positions - self._codes



        # Update velocities towards the personal best positions

        self.pso_optimizer.velocity = self.pso_optimizer.inertia * self.pso_optimizer.velocity + self.pso_optimizer.cognitive * self._rng.uniform(size=(self.pop_size, self.codes_len)) * position_deltas 

        #clamping the velocity

        self.pso_optimizer.velocity = np.clip(

            self.pso_optimizer.velocity, -self.pso_optimizer.v_clip, self.pso_optimizer.v_clip

        )

        

        return self.pso_optimizer.velocity



    def _adapt_hyperparameters(self):

        

        recent_mean_fitness = self.history['mean_fitness'][-15:]

        

        

        # improvement rates based on recent history

        mean_improvement = np.mean(np.diff(recent_mean_fitness))

        diversity = np.std(self._codes, axis=0).mean()

        

        

        # is_stagnating is a bool variable that checks if the fitness is stagnating

        is_stagnating = (abs(mean_improvement) < self.stagnation_threshold) 

        

        # Dynamic parameter adjustment

        if is_stagnating and diversity < self.diversity_threshold: 

            

             

            #self.pso_optimizer.cognitive *= self.less_cog_factor

            self.ga_optimizer._mut_rate = min(0.8, self.ga_optimizer._mut_rate * self.more_mut_rate)

                

                

        else:

            

            self.pso_optimizer.cognitive *= self.more_cog_factor  # Increase exploitation

            # If the fitness is chnaging constantly then reduce the mutation rate and size

            self.ga_optimizer._mut_rate = max(0.3, self.ga_optimizer._mut_rate * self.less_mut_rate)

            self.ga_optimizer._mut_size = max(0.1, self.ga_optimizer._mut_size * self.less_mut_size)





    def _step(self, scores: Fitness) -> Codes:

        # Stepwise scheduler for PSO frequency

        progress_ratio = self.current_iteration / self.max_iterations

        

        

        self.ga_optimizer._codes = self._codes.copy()

        self._codes = self.ga_optimizer._step(scores)

    

        self._reset_velocity_after_ga()

        self._update_pso_after_ga(scores)                       ## APPARETLY THIS YIELDS BETTER RESULTS



        

        # If < 0.3 of total iterations, do mostly GA

        # if >= 0.3 and < 0.8, a moderate frequency

        # if >= 0.8, do it more frequently for fine-tuning



        if progress_ratio < self.first_PSO_interval:

            PSO_interval = 5           #5


        elif progress_ratio < self.second_PSO_interval:

            PSO_interval = 2            # 2

          

        else:

            PSO_interval = 1            #2

      

        

        

        # Apply PSO more frequently as iterations progress

        if self.current_iteration % PSO_interval == 0 :

            

           

            # Compare current mean scores to recent history to ensure improvement

            #if np.mean(scores) > np.mean(self.history['mean_fitness'][-10:]):

                

                

                # Copy current population to PSO optimizer

            self.pso_optimizer._codes = self._codes.copy()
            # Execute PSO step
            self.pso_optimizer._step(scores, update_inertia=True)
            # Update main population with PSO results
            self._codes = self.pso_optimizer._codes.copy()




            if np.mean(scores) > np.mean(self.history['mean_fitness'][-3:]):
                 
                self._adapt_hyperparameters()

        

     

        

        

        

     

        self.history['mean_fitness'].append(np.mean(scores))

        self.history['best_fitness'].append(np.max(scores))
     

        self.current_iteration += 1

        

        return self._codes
