from collections import OrderedDict
from ophyd.signal import (Signal, EpicsSignal, EpicsSignalRO)
from ophyd import Device, PVPositioner, PVPositionerPC
from ophyd import (Component as C, DynamicDeviceComponent as DDC)

import time as ttime
from bluesky.protocols import NamedMovable, Readable, Status, Hints, HasHints, HasParent
from typing import Any
import pandas as pd
import scipy as sp
import cv2
from unittest.mock import MagicMock

from ophyd import Device, EpicsSignal
from ophyd import Component as Cpt
from ophyd.sim import instantiate_fake_device

from bluesky.utils import MsgGenerator
import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky import RunEngine
import numpy as np
from tiled.client.container import Container

from blop import DOF, Objective, Agent, DOFConstraint
from blop.ax import Agent as AxAgent
from blop.plans import acquire
import blop.data_access

class AlwaysSuccessfulStatus(Status):
    def add_callback(self, callback) -> None:
        callback(self)

    def exception(self, timeout = 0.0):
        return None
    
    @property
    def done(self) -> bool:
        return True
    
    @property
    def success(self) -> bool:
        return True

class ReadableSignal(Readable, HasHints, HasParent):
    def __init__(self, name: str) -> None:
        self._name = name
        self.cam = MagicMock()
        self.cam.name = name
        self._value = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def hints(self) -> Hints:
        return { 
            "fields": [self._name],
            "dimensions": [],
            "gridding": "rectilinear",
        }
    
    @property
    def parent(self) -> Any | None:
        return None

    def read(self):
        return {
            self._name: { "value": self._value, "timestamp": ttime.time() }
        }

    def describe(self):
        return {
            self._name: { "source": self._name, "dtype": "number", "shape": [] }
        }

class MovableSignal(ReadableSignal, NamedMovable):
    def __init__(self, name: str, initial_value: float = 0.0) -> None:
        super().__init__(name)
        self._value: float = initial_value

    def set(self, value: float) -> Status:
        self._value = value
        return AlwaysSuccessfulStatus()

class Channel(PVPositionerPC):
    '''Bimorph Channel'''
    setpoint = C(EpicsSignal, '_SP.VAL')
    readback = C(EpicsSignalRO, '_CURRENT_MON_VAL.VAL')
    armed_voltage = C(EpicsSignalRO, '_SP_MON.VAL')
    #target_voltage = C(EpicsSignalRO, '_TARGET_MON_VAL.VAL')
    #min_voltage = C(EpicsSignalRO, '_MINV_MON.VAL')
    #max_voltage = C(EpicsSignalRO, '_MAXV_MON.VAL')

    # done = Cpt(Signal, value=0)

    # done_value = 0

    # def read(self):

    #     res = super().read()
    #     for key in res.keys():
    #         value = res[key]["value"]
    #         value = np.atleast_1d(value)
    #         if len(value) > 0:
    #             value = value[0]
    #         res[key]["value"] = value

    #     return res

    # def describe(self):

    #     res = super().describe()

    #     for key in res.keys():

    #         res[key]["shape"] = []
    #         res[key]["dtype"] = "integer" if "done" in key else "number"

    #     return res


    #done = Cpt(EpicsSignalRO, 'Cmd-Busy')
    #stop_signal = Cpt(EpicsSignal, 'Cmd-Cmd')


    # def set(self, value):

    #     # if (value < self.min_voltage.get()):
    #     #     raise ValueError("Desired voltage is too low!")
    #     # if (value > self.max_voltage.get()):
    #     #     raise ValueError("Desired voltage is too high!")



    #     return st

    # def get(self):

    #     return self.setpoint.get()


def add_channels(range_, **kwargs):
    '''Add one or more Channel to an Bimorph instance
       Parameters:
       -----------
       range_ : sequence of ints
           Must be be in the set [0,31]
       By default, an Bimorph is initialized with all 32 channels.
       These provide the following Cs as EpicsSignals (N=[0,31]):
       Bimorph.channels.channelN.(fields...)
       '''
    defn = OrderedDict()

    for ch in range_:
        if not (0 <= ch < 32):
            raise ValueError('Channel must be in the set [0,31]')

        attr = 'channel{}'.format(ch)
        defn[attr] = (Channel, ':U{}'.format(ch), kwargs)


    return defn

class Bimorph(Device):
    '''Bimorph HV Power Source'''

    bank_no = C(EpicsSignal, ':BANK_NO_32.VAL')
    step_size = C(EpicsSignal, ':U_STEP.VAL')
    inc_bank = C(EpicsSignal, ':INCR_U_BANK_CMD.PROC')
    dec_bank = C(EpicsSignal, ':DECR_U_BANK_CMD.PROC')
    stop_ramp = C(EpicsSignal, ':STOP_RAMPS_BANK.PROC')
    start_ramp = C(EpicsSignal, ':START_RAMPS_CMD.PROC')

    format_number = C(EpicsSignal, ':FORMAT_NO_SP.VAL')
    load_format = C(EpicsSignal, ':FORMAT_ACTIVE_SP.PROC')

    all_target_voltages = C(EpicsSignalRO, ':U_ALL_TARGET_MON.VAL')
    all_current_voltages = C(EpicsSignalRO, ':U_ALL_CURRENT_MON.VAL')

    unit_status = C(EpicsSignalRO, ':UNIT_STATUS_MON.A')

    channels = DDC(add_channels(range(0, 32)))

    def step(self, bank, size, direction, start=False, wait=False):
        self.bank_no.put(bank)
        self.step_size.put(size)

        if(direction == "inc"):
            self.inc_bank.put(1)
        else:
            self.dec_bank.put(1)

        if(start):
            self.start()

        if(wait):
            self.wait()

    def increment_bank(self, bank, size, start=False, wait=False):
        ''' Increments the target voltage in `size` Volts in the specified `bank`

        Parameters:
        -----------
        bank : int
            The number of the bank to be incremented
        size : float
            The amount of Volts to increment from the bank target value
        start : bool
            Determines if the ramp must start right after the increment. Defaults to False.
        wait : bool
            Determines if the code must wait until the ramp process finishes. Defaults to False.
        '''
        self.step(bank, size, "inc", start)

    def decrement_bank(self, bank, size, start=False, wait=False):
        ''' Decrements the target voltage in `size` Volts in the specified `bank`

        Parameters:
        -----------
        bank : int
            The number of the bank to be decremented
        size : float
            The amount of Volts to decrement from the bank target value
        start : bool
            Determines if the ramp must start right after the decrement. Defaults to False.
        wait : bool
            Determines if the code must wait until the ramp process finishes. Defaults to False.
        '''
        self.step(bank, size, "dec", start)

    def start(self):
        ''' Start the Ramping process on all channels '''
        self.start_ramp.put(1)

    def stop(self):
        ''' Stops the Ramping process on all channels '''
        self.stop_ramp.put(1)

    def start_plan(self):
        yield from bps.mv(self.start_ramp, 1)

    def is_ramping(self):
        ''' Returns wether the power supply is ramping or not '''
        return (int(self.unit_status.get()) >> 30) == 1

    def is_interlock_ok(self):
        ''' Returns the interlock state '''
        st = int(self.unit_status.get())
        return (st & 1) & ((st >> 1) & 1) == 1

    def is_on(self):
        ''' Returns wether the Channels are ON or OFF '''
        return (int(self.unit_status.get()) >> 29) == 1          

    def wait(self):
        while self.is_ramping():
          sleep(0.1)

    def all_armed_voltages(self):
        return np.array([getattr(self, f"channels.channel{i}.armed_voltage").get() for i in range(32)], dtype=np.float32).ravel()

    def all_setpoint_voltages(self):
        return np.array([getattr(self, f"channels.channel{i}.setpoint").get() for i in range(32)], dtype=np.float32).ravel()


RE = RunEngine({})
bimorph = instantiate_fake_device(Bimorph, name="bimorph")
scnSS = ReadableSignal(name="scnSS")
db = MagicMock(spec=Container)
blop.data_access.TiledDataAccess.get_data = MagicMock(return_value={"scnSS_image": [np.random.randint(0, 255, (100, 100)).astype(np.uint16)]})
def mock_move_per_step(step, pos_cache):
    yield from bps.null()
bps.move_per_step = mock_move_per_step

import numpy as np
import scipy as sp


def vertical_profile_metric(image, background=None, threshold_factor=0.1,
                           intensity_weight=1.0, uniformity_weight=1.0,
                           edge_crop=0):
    """
    Metric for vertical beam uniformity optimization (for vertical focusing mirror).
    Collapses image to 1D vertical profile and optimizes for uniformity + intensity.
    
    Lower values = better (minimize this metric).
    
    :param image: OpenCV image (BGR or grayscale)
    :param background: Optional background image for subtraction
    :param threshold_factor: Fraction of max intensity for beam detection (default 0.1)
    :param intensity_weight: Weight for intensity maximization term
    :param uniformity_weight: Weight for vertical uniformity term
    :param edge_crop: Number of pixels to crop from the edges of the image. Default to 0.
    :return: Tuple (metric: float, debug_image: OpenCV image, metrics_dict: dict)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape) == 2 else image.copy()

    # Crop edges to remove artifacts
    if edge_crop > 0:
        gray = gray[edge_crop:-edge_crop, edge_crop:-edge_crop]
        if background is not None:
            background = background[edge_crop:-edge_crop, edge_crop:-edge_crop]
    
    # Background subtraction
    if background is None:
        background = np.zeros_like(gray)
    else:
        if len(background.shape) == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    corrected = cv2.subtract(gray, background)
    corrected = cv2.GaussianBlur(corrected, (5, 5), 0)
    
    max_intensity = np.max(corrected)
    if max_intensity == 0:
        return float('inf'), debug_img, {}
        
    thresh_value = threshold_factor * max_intensity
    _, thresh = cv2.threshold(corrected, thresh_value, 255, cv2.THRESH_BINARY)
    
    # ========== VERTICAL PROFILE ==========
    # Collapse to 1D vertical profile by averaging over horizontal (x) axis
    vertical_profile = np.mean(thresh, axis=1)  # Average over horizontal direction
    
    if len(vertical_profile) == 0 or np.sum(vertical_profile) == 0:
        return float('inf'), debug_img, {}
    
    # ========== UNIFORMITY METRIC ==========
    # Coefficient of variation (CV) = std / mean
    mean_intensity = np.mean(vertical_profile)
    std_intensity = np.std(vertical_profile)
    cv = std_intensity / mean_intensity if mean_intensity > 0 else float('inf')
    
    # ========== TOTAL INTENSITY ==========
    # Total integrated intensity (sum of profile)
    total_intensity = np.sum(vertical_profile)
    
    # ========== COMBINED METRIC ==========
    # Minimize CV (uniformity) and maximize intensity (negate for minimization)
    metric = uniformity_weight * cv - intensity_weight * total_intensity
    
    # ========== DEBUG VISUALIZATION ==========
    # Plot vertical profile to the right of the image
    profile_width = 150
    profile_x_start = debug_img.shape[1] - profile_width - 10
    
    # Normalize profile for plotting
    if np.max(vertical_profile) > 0:
        normalized_profile = (vertical_profile / np.max(vertical_profile)) * profile_width
    else:
        normalized_profile = np.zeros_like(vertical_profile)
    
    # Draw profile graph (rotated - grows to the right)
    # Profile spans full image height (0 to image height)
    for i in range(len(normalized_profile) - 1):
        pt1 = (profile_x_start + int(normalized_profile[i]), i)
        pt2 = (profile_x_start + int(normalized_profile[i + 1]), i + 1)
        cv2.line(debug_img, pt1, pt2, (255, 255, 0), 2)
    
    # Draw baseline for profile (full height)
    cv2.line(debug_img, (profile_x_start, 0), (profile_x_start, debug_img.shape[0]), (255, 255, 255), 1)
    
    # Text overlays
    text_x = 10
    cv2.putText(debug_img, f"CV (Uniformity): {cv:.4f}", 
                (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Total Intensity: {total_intensity:.1f}", 
                (text_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Mean: {mean_intensity:.1f}", 
                (text_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Std Dev: {std_intensity:.1f}", 
                (text_x, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Metric: {metric:.4f}", 
                (text_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    metrics_dict = {
        'cv': cv,
        'total_intensity': total_intensity,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'metric': metric,
        'vertical_profile': vertical_profile
    }
    
    return metric, debug_img, metrics_dict


def vertical_profile_digestion(
    trial_index: int,
    readings: dict[str, list[Any]],
    threshold_factor: float = 0.1,
    edge_crop: int = 0,
) -> dict[str, float | tuple[float, float]]:
    """
    Digestion function for vertical profile optimization.

    Parameters
    ----------
    trial_index : int
        The index of the trial.
    readings : dict[str, list[Any]]
        The readings from the optimization detectors.
    threshold_factor : float, optional
        The factor to multiply the maximum intensity by to get the threshold for the vertical profile. Default to 0.1.
    edge_crop : int, optional
        The number of pixels to crop from the edges of the image. Default to 0.
    """
    image = readings[f"{scnSS.cam.name}_image"][0]
    _, _, metrics_dict = vertical_profile_metric(image, threshold_factor=threshold_factor, edge_crop=edge_crop)
    print(f"Metrics dict: {metrics_dict}")
    return {
        "vertical_coefficient_variation": metrics_dict["cv"].item(),
        "total_vertical_intensity": metrics_dict["total_intensity"].item(),
    }


def _get_channel_neighbors(channel: int) -> list[Channel]:
    """Helper function to get the neighbors of a channel in the bimorph."""
    neighbors_indices = []
    # Horizontal mirror channels: 0-11
    if 0 <= channel <= 11:
        if channel == 0:
            neighbors_indices.append(1)
        elif channel == 11:
            neighbors_indices.append(10)
        else:
            neighbors_indices.append(channel - 1)
            neighbors_indices.append(channel + 1)
    
    # Vertical mirror channels: 12-23
    elif 12 <= channel <= 23:
        if channel == 12:
            neighbors_indices.append(13)
        elif channel == 23:
            neighbors_indices.append(22)
        else:
            neighbors_indices.append(channel - 1)
            neighbors_indices.append(channel + 1)
    
    # Unknown channels: 24-31
    else:
        return []
    
    return [
        getattr(bimorph.channels, f"channel{i}")
        for i in neighbors_indices
    ]


def _setup_bimorph_dofs(channel_range: range, search_radius: float = 100.0, constraint: float | None = 300.0):
    """
    Sets up the DOFs for the bimorph given a range of channels.

    Parameters
    ----------
    channel_range : range
        The range of channels to setup the DOFs for. 0-11 are horizontal mirror, 12-23 are vertical mirror, 24-31 are unknown.
    search_radius : float, optional
        How wide the search domain should be for each DOF. Default to 100.0 V.
    constraint : float | None, optional
        Constrain the search space such that each channel's distance from its neighbor is within the constraint. Default is
        300.0 V. If None, no constraint is applied. This is important for safety of the bimorph.

    Returns
    -------
    dofs : list[DOF]
        The degrees of freedom for the bimorph.
    dof_constraints : list[DOFConstraint]
        The constraints on the degrees of freedom.
    """

    dofs = []
    dof_constraints = []
    for channel in channel_range:
        bimorph_channel = getattr(bimorph.channels, f"channel{channel}")
        current_pos = bimorph_channel.readback.get()
        dofs.append(
            DOF(
                movable=bimorph_channel,
                type="continuous",
                search_domain=(current_pos - search_radius, current_pos + search_radius),
            )
        )
        if constraint is not None:
            # Apply distance constraints between neighbor channels
            # Blop only supports linear constraints, so for a distance constraint, we need to apply two separate constraints.
            neighbors = _get_channel_neighbors(channel)
            for neighbor in neighbors:
                dof_constraints.append(
                    DOFConstraint(f"x1 - x2 <= {constraint}", x1=bimorph_channel, x2=neighbor)
                )
                dof_constraints.append(
                    DOFConstraint(f"x2 - x1 <= {constraint}", x1=bimorph_channel, x2=neighbor)
                )
    return dofs, dof_constraints


vertical_mirror_dofs, vertical_mirror_dof_constraints = _setup_bimorph_dofs(
    range(12, 24),
    search_radius=100.0,
    constraint=300.0,
)
uniform_vertical_profile_objectives = [
    Objective(name="vertical_coefficient_variation", target="min"),
    Objective(name="total_vertical_intensity", target="max"),
]
optimization_detectors = [scnSS]
uniform_vertical_profile_agent = AxAgent(
    readables=optimization_detectors,
    dofs=vertical_mirror_dofs,
    objectives=uniform_vertical_profile_objectives,
    db=db,
    dof_constraints=vertical_mirror_dof_constraints,
    digestion=vertical_profile_digestion,
    digestion_kwargs={"threshold_factor": 0.1, "edge_crop": 0},
)


def one_nd_bimorph_step(detectors, step, pos_cache, take_reading=None, *, bimorph_device=None, timeout=10, tolerance=1, poll_interval=0.1):
    """
    Per-step hook that allows for coordinated movement of the bimorph mirrors.
    
    The bimorph mirror system requires that the actuators be armed, then ramped to the target voltages.
    Setting an individual channel (actuator) only sets the target voltage for that channel.

    Parameters
    ----------
    detectors : list[Readable]
        The detectors to take a reading from
    step : dict[NamedMovable, Any]
        The next position to move to for each movable device
    pos_cache : dict[NamedMovable, Any]
        The last position moved to for each movable device
    take_reading : Callable[[list[Readable]], MsgGenerator] | None, optional
        Custom plan hook to take a reading from the detectors, defaults to ``bps.trigger_and_read``
    bimorph : Bimorph | None, optional
        The bimorph mirror system to control, default to ``bimorph`` device defined in the namespace
    timeout : float, optional
        The timeout for the bimorph to arm and ramp, default to 10 seconds
    tolerance : float, optional
        The tolerance for each channel's residual voltage to be within, default to 1 V
    poll_interval : float, optional
        The interval to poll the bimorph to arm and ramp, default to 0.001 seconds
    """

    if take_reading is None:
        take_reading = bps.trigger_and_read
    print("[one_nd_bimorph_step] take_reading determined")

    if bimorph_device is None:
        bimorph_device = bimorph
    print("[one_nd_bimorph_step] bimorph_device determined")

    def _armed() -> bool:
        armed_voltages = np.array(bimorph_device.all_armed_voltages(), dtype=np.float32)
        return np.allclose(armed_voltages, bimorph_device.all_setpoint_voltages(), atol=tolerance)

    def _ramped() -> bool:
        current_voltages = np.array(bimorph_device.all_current_voltages.get(), dtype=np.float32)
        return np.allclose(current_voltages, bimorph_device.all_setpoint_voltages(), atol=tolerance)
    print("[one_nd_bimorph_step] _armed and _ramped checkers defined")
    
    # Move to the next position (change setpoints for bimorph channels)
    print("[one_nd_bimorph_step] Moving to next position (setpoints for bimorph channels)")
    yield from bps.move_per_step(step, pos_cache)

    # Wait for the bimorph mirror to be armed
    print("[one_nd_bimorph_step] Waiting for bimorph mirror to be armed...")
    start_time = ttime.monotonic()
    while not _armed():
        yield from bps.sleep(poll_interval)
        if ttime.monotonic() - start_time > timeout:
            raise TimeoutError(f"Failed to arm the bimorph mirrors within {timeout} seconds")

    # Start ramping
    print("[one_nd_bimorph_step] Starting ramping")
    yield from bimorph_device.start_plan()

    # Wait for the bimorph mirror to be ramped
    print("[one_nd_bimorph_step] Waiting for bimorph mirror to be ramped...")
    start_time = ttime.monotonic()
    while not _ramped():
        yield from bps.sleep(poll_interval)
        if ttime.monotonic() - start_time > timeout:
            raise TimeoutError(f"Failed to ramp the bimorph mirrors within {timeout} seconds")

    # settle time after ramping
    print("[one_nd_bimorph_step] Settling after ramping...")
    yield from bps.sleep(1.0)

    # Take a reading from the detectors
    print("[one_nd_bimorph_step] Taking a reading from detectors and step movers...")
    yield from take_reading(list(detectors) + list(step.keys()))


def optimize_vertical_profile(iterations: int = 30) -> MsgGenerator[None]:
    """
    Optimization plan to optimize the vertical beam profile to maximize intensity and minimize coefficient of variation.
    """
    for _ in range(iterations):
        trials = uniform_vertical_profile_agent.get_next_trials(1)
        uid = yield from acquire(
            uniform_vertical_profile_agent.readables,
            uniform_vertical_profile_agent.dofs,
            trials,
            per_step=one_nd_bimorph_step,
        )
        results = uniform_vertical_profile_agent.data_access.get_data(uid)
        data = {trial_index: uniform_vertical_profile_agent.digestion(trial_index, results, **uniform_vertical_profile_agent.digestion_kwargs) for trial_index in trials.keys()} 
        uniform_vertical_profile_agent.complete_trials(data)
