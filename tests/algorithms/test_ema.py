import copy
import pytest
import numpy as np
import torch

from composer.algorithms import EMAHparams
from composer.algorithms.ema.ema import ema
from composer.core import Event, Time, Timer, TimeUnit
from tests.common import SimpleModel, SimpleConvModel


def validate_ema(model, original_model, ema_model, decay):
    model_dict = model.state_dict()
    original_dict = original_model.state_dict()
    ema_dict = ema_model.state_dict()

    for key, param in original_dict.items():
        model_param = model_dict[key].detach()
        new_param = param * decay + (1. - decay) * model_param
        torch.testing.assert_allclose(ema_dict[key], new_param)


@pytest.mark.parametrize("decay", [0, 0.5, 0.99, 1])
def test_ema(decay):
    model = SimpleModel()
    ema_model = SimpleModel()
    original_model = copy.deepcopy(ema_model)
    ema(model=model, ema_model=ema_model, decay=decay)
    validate_ema(model, original_model, ema_model, decay)


# params = [(half_life, update_interval)]
@pytest.mark.parametrize('params', [("10ba", "1ba"), ("1ep", "1ep")])
def test_ema_algorithm(params, minimal_state, empty_logger):

    # Initialize input tensor
    input = torch.rand((32, 5))

    half_life, update_interval= params[0], params[1]
    algorithm = EMAHparams(half_life=half_life, update_interval=update_interval, train_with_ema_weights=False).initialize_object()
    state = minimal_state
    state.model = torch.nn.Linear(5, 5)
    state.batch = (input, torch.Tensor())

    # Start EMA
    algorithm.apply(Event.FIT_START, state, empty_logger)
    # Check if ema correctly calculated decay
    half_life = Time.from_timestring(params[0])
    update_interval = Time.from_timestring(params[1])
    decay = np.exp(-np.log(2) * (update_interval.value/half_life.value))
    np.testing.assert_allclose(decay, algorithm.decay)

    # Fake a training update by replacing state.model after ema grabbed it.
    original_model = copy.deepcopy(state.model)
    state.model = torch.nn.Linear(5, 5)
    # Do the EMA update
    state.timer = Timer()
    if half_life.unit == TimeUnit.BATCH:
        state.timer._batch = update_interval
        algorithm.apply(Event.BATCH_END, state, empty_logger)
    elif half_life.unit == TimeUnit.EPOCH:
        state.timer._epoch = update_interval
        algorithm.apply(Event.EPOCH_END, state, empty_logger)
    else:
        raise ValueError(f"Invalid time string for parameter half_life")
    # Check if EMA correctly computed the average.
    validate_ema(state.model, original_model, algorithm.ema_model, algorithm.decay)
    # Check if the EMA model is swapped in for testing
    algorithm.apply(Event.EVAL_START, state, empty_logger)
    # Check if the training model is swapped back in for training
    algorithm.apply(Event.EVAL_END, state, empty_logger)
