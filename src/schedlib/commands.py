from typing import List
from dataclasses import dataclass, field
from datetime import datetime
import inspect

from . import core


@dataclass(frozen=True)
class Operation:
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
        A tuple containing an estimated duration and a list of command strings.

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
    def __call__(self, state):
        return 0, []

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

        class _Operation(Operation):
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
            def __call__(self, state):
                # decide whether duration is provided or will be computed
                # as part of the operation function
                if alter_state:
                    res = operation_fun(state, *self.args, **self.kwargs)
                else:
                    res = operation_fun(*self.args, **self.kwargs)

                if return_duration:
                    return res
                else:
                    res = (duration, res)
                return res

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

class OperationBlock(core.NamedBlock):
    commands: List[str]
