import datetime as dt
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, replace as dc_replace
from abc import ABC, abstractmethod
import inspect

from . import core

MIN_DURATION = 0.01


class SchedMode:
    """
    Enumerates the scheduling modes for satellite operations.

    Attributes
    ----------
    PreCal : str
        'pre_cal'; Operations scheduled before block.t0 for calibration.
    PreObs : str
        'pre_obs'; Observations scheduled before block.t0 for observation.
    InCal : str
        'in_cal'; Calibration operations scheduled between block.t0 and block.t1.
    InObs : str
        'in_obs'; Observation operations scheduled between block.t0 and block.t1.
    PostCal : str
        'post_cal'; Calibration operations scheduled after block.t1.
    PostObs : str
        'post_obs'; Observations operations scheduled after block.t1.
    PreSession : str
        'pre_session'; Represents the start of a session, scheduled from the beginning of the requested t0.
    PostSession : str
        'post_session'; Indicates the end of a session, scheduled after the last operation.

    """
    PreCal = 'pre_cal'
    PreObs = 'pre_obs'
    InCal = 'in_cal'
    InObs = 'in_obs'
    PostCal = 'post_cal'
    PostObs = 'post_obs'
    PreSession = 'pre_session'
    PostSession = 'post_session'


@dataclass(frozen=True)
class State:
    """
    A dataclass representing the state of an observatory at a specific time. It should
    contain all relevant information for planning operations,

    Parameters
    ----------
    curr_time : datetime.datetime
        The current timestamp of the state.
    az_now : float
        The current azimuth position in degrees.
    el_now : float
        The current elevation position in degrees.
    az_speed_now : Optional[float], optional
        The current azimuth speed in degrees per second. Default is None.
    az_accel_now : Optional[float], optional
        The current azimuth acceleration in degrees per second squared. Default is None.
    prev_state : Optional[State], optional
        A reference to the previous state for tracking state evolution. Default is None and not included in string representation.

    Methods
    -------
    replace(**kwargs)
        Returns a new State instance with the specified attributes replaced with new values.
    increment_time(dt)
        Returns a new State instance with the current time incremented by a datetime.timedelta.
    increment_time_sec(dt_sec)
        Returns a new State instance with the current time incremented by a specific number of seconds.

    """
    curr_time: dt.datetime
    az_now: Optional[float] = None
    el_now: Optional[float] = None
    az_speed_now: Optional[float] = None
    az_accel_now: Optional[float] = None
    prev_state: Optional["State"] = field(default=None, repr=False)

    def replace(self, **kwargs):
        """
        Creates a new instance of the State class with specified attributes replaced with new values,
        while automatically setting 'prev_state' to the current state to maintain history.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments corresponding to the attributes of the State class which are to be replaced.

        Returns
        -------
        State
            A new instance of the State class with updated attributes.

        Examples
        --------
        >>> new_state = old_state.replace(az_now=180)
        """
        kwargs = {**kwargs, "prev_state": self}
        return dc_replace(self, **kwargs)

    def increment_time(self, dt: dt.timedelta) -> "State":
        """
        Returns a new State instance with the current time incremented by a specified datetime.timedelta.

        Parameters
        ----------
        dt : datetime.timedelta
            The amount of time to increment the current time.

        Returns
        -------
        State
            A new State instance with the incremented current time.
        """
        return self.replace(curr_time=self.curr_time+dt)

    def increment_time_sec(self, dt_sec: float) -> "State":
        """
        Increments the current time of the State by a specified number of seconds
        and returns a new State instance with this updated time.

        Parameters
        ----------
        dt_sec : float
            The number of seconds to increment the current time by.

        Returns
        -------
        State
            A new State instance with the current time incremented by the specified number of seconds.
        """
        return self.replace(curr_time=self.curr_time+dt.timedelta(seconds=dt_sec))


@dataclass(frozen=True)
class Operation(ABC):
    """
    An operation is a callable object that produces a set of commands to be executed by the observatory.

    This block takes an observatory state (dict) as input and returns a tuple containing a new state object
    as the effect of the command, and a list of commands for the telescope. Each command in the list is a 
    string-like object.

    Parameters
    ----------
    state : object
        The current state of the observatory.

    Returns
    -------
    tuple
        A tuple containing a new state, an estimated duration, and a list of command strings.

    Notes
    -----
    This is a base class for all operations. It should not be used directly.
    state will be mutated by the operation in place, therefore no "parallelism"
    is allowed on the same state object.

    Examples
    --------
    op = Operation()
    state = get_init_state()
    duration, commands = op(state)

    """
    @abstractmethod
    def __call__(self, state: State) -> Tuple[State, float, List[str]]:
        ...

OPERATION_REGISTRY = {}

def register_operation_cls(name, operation_cls):
    """
    Register a new operation in the operation registry.

    Parameters
    ----------
    name : str
        The name of the operation to register.
    operation_cls : class
        The class of the operation to register.

    Returns
    -------
    class
        The class of the operation that was registered.

    """
    if name in OPERATION_REGISTRY:
        raise ValueError(f"Operation {name} already exists in the registry.")
    OPERATION_REGISTRY[name] = operation_cls
    return operation_cls

def get_operation_cls(name):
    """
    Get an operation class from the operation registry.

    Parameters
    ----------
    name : str
        The name of the operation to retrieve.

    Returns
    -------
    class
        The class of the operation that was retrieved.

    """
    if name not in OPERATION_REGISTRY:
        raise ValueError(f"Operation {name} does not exist in the registry.")
    return OPERATION_REGISTRY[name]

def operation_cls(name):
    """
    A decorator that registers an operation class in the operation registry.

    Parameters
    ----------
    name : str
        The name of the operation to register.

    Returns
    -------
    class
        The class of the operation that was registered.

    """
    def wrapper(operation_cls):
        return register_operation_cls(name, operation_cls)
    return wrapper

def operation(name, duration=0, return_duration=False):
    """
    A decorator that registers a function as an operation
    in the operation registry.

    Parameters
    ----------
    name : str
        The name of the operation to register.

    Returns
    -------
    class
        The class of the operation that was registered.

    """
    def wrapper(operation_fun):
        # check whether the operation function wants to update the state dict
        sig = inspect.signature(operation_fun)
        args = list(sig.parameters.keys())
        alter_state = len(args) > 0 and args[0] == "state"

        # sometimes the block is automatically passed in for
        # per-block operations, but the operation may not needed it.
        # here we will explicitly remove it from the kwargs if it is
        # not in the original function signature
        want_block = "block" in args

        class _Operation(Operation):
            def __init__(self, *args, **kwargs):
                self.args = args
                if not want_block: kwargs.pop("block", None)
                self.kwargs = kwargs
            def __call__(self, state):
                # decide whether duration is provided or will be computed
                # as part of the operation function
                _duration = duration
                if alter_state:
                    state, *rest = operation_fun(state, *self.args, **self.kwargs)
                    if len(rest) == 1: rest = rest[0]
                else:
                    rest = operation_fun(*self.args, **self.kwargs)

                if return_duration:
                    _duration, commands = rest
                else:
                    commands = rest

                # avoid 0 duration to prevent sorting problems
                _duration = max(MIN_DURATION, _duration)
                return state, _duration, commands

        return register_operation_cls(name, _Operation)

    return wrapper

def make_op(name, *args, **kwargs):
    """
    Make an operation object from the operation registry.

    Parameters
    ----------
    name : str
        The name of the operation to create.
    *args : list
        The arguments to pass to the operation constructor.
    **kwargs : dict
        The keyword arguments to pass to the operation constructor.

    Returns
    -------
    Operation
        An operation object.

    """
    op_cls = get_operation_cls(name)
    return op_cls(*args, **kwargs)

@dataclass(frozen=True)
class OperationBlock(core.NamedBlock):
    subtype: Optional[str] = None
    commands: List[str] = field(default_factory=list, repr=False)
    parameters: Dict = field(default_factory=dict, repr=False)

    def __hash__(self):
        return hash((self.name, self.t0, self.t1, self.subtype))
