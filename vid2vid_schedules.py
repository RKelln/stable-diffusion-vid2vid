
from typing import Any, Callable

# parse a string time in the form of 00:00:00.000 to seconds
def time_string_to_seconds(time_str : str) -> float:
    time_str = time_str.strip()
    if time_str == "":
            return 0.0

    time_parts = time_str.split(':')
    if len(time_parts) == 0:
        return float(time_str)
    
    h = m = s = s_ms = 0.0
    if len(time_parts) == 1:
        s_ms = float(time_parts[0])
    elif len(time_parts) == 2:
        m, s_ms = map(float, time_parts)
    else:
        h, m, s_ms = map(float, time_parts)
    s, ms = map(float, f"{s_ms:0.03f}".split('.'))
    if s < 0:
        raise RuntimeError(f"Invalid time seconds: {time_str}")
    if m >= 60.0 or m < 0:
        raise RuntimeError(f"Invalid time minutes: {time_str}")
    if h < 0:
        raise RuntimeError(f"Invalid time hours: {time_str}")
    return 3600*h + 60*m + s + ms/1000.0

# given a string of time or frame, return time in seconds
def parse_to_seconds(s : str, fps : float, duration : float) -> float:
    if fps <= 0:
        raise RuntimeError(f"FPS must be > 0: {fps}")
    if duration <= 0:
        raise RuntimeError(f"Duration must be > 0: {duration}")

    s = s.strip()
    if s == "":
        return 0.
    elif s.endswith('%'):
        percentage = max(0.0, min(100.0, float(s.strip('%'))))
        return (percentage / 100.) * duration
    elif ':' in s:
        return max(0.0, min(duration, time_string_to_seconds(s)))
    elif '.' in s:
        return max(0.0, min(duration, float(s)))
    else: # integers treated as frames
        return max(0.0, min(duration, frame_to_time(int(s), fps)))


def parse_schedule(s : str, fps : float, duration : float, default_value : Any) -> list:
    values = []
    s = s.strip()
    if s != "":
        value_strs = [x.strip() for x in s.split(",")]
        for v_str in value_strs:
            if v_str.strip() == "": continue
            time_value_split = v_str.rsplit(":", maxsplit=1)
            t = parse_to_seconds(time_value_split[0], fps, duration)
            val = time_value_split[1].strip(" ()")
            values.append((t, val))
    if len(values) == 0:
        values.append((0, default_value))
    return sorted(values)

# Frames count from 1 and the start of a frame is time 0
def frame_to_time(frame : int, fps : float) -> float:
    if frame < 1:
        frame = 1 # assume they met the first frame
    return round(float(frame - 1) / fps, 3) # round to millseconds

# Frames count from 1
def time_to_frame(time : float, fps : float) -> int:
    # compensate for floats being weird
    # at 30 fps, first frame from 0 to 32ms, second frame starts at 33ms
    # e.g. 333ms = 0.333 * 30 fps = 9.99, we want to return frame 10
    return int( time * fps ) + 1


def seed_travel_planning(seeds : list, frame_count : int, fps : float, starting_seed : int) -> list:

    def seed_values(v, next_v, step, steps) -> tuple:
        if next_v == None or next_v == v:
            return (v, 0, 0)
        else:
            strength = float(step)/float(steps)
            if strength == 0.0:
                return (v, 0, 0)
            elif strength == 1.0: # shouldn't happen
                return (next_v, 0, 0)
            else:
                return (v, next_v, strength)

    return travel_planning(seeds, frame_count, fps, seed_values, starting_seed)


def basic_lerp(v, next_v, step, steps) -> float:
        if next_v == None or next_v == v:
            return v
        else:
            strength = float(step)/float(steps)
            if strength == 0.0:
                return v
            elif strength == 1.0:
                return next_v
            else:
                return v + (strength * (next_v - v))


def denoise_travel_planning(noise_timings : list, frame_count : int, fps : float, starting_noise : float) -> list:

    return travel_planning(noise_timings, frame_count, fps, basic_lerp, starting_noise)


# value_fn(v, next_v, step, steps) -> tuple:
def travel_planning(values : list, frame_count : int, fps : float, value_fn: Callable[[Any, Any, int, int], Any], starting_value : any) -> list:
    values_per_frame = []
    current_frame = frame = 0
    v = next_v = starting_value
    next_frame = frame_count

    # NOTE: frames count from 1, but list is 0 indexed
    while len(values_per_frame) < frame_count:
        current_frame = len(values_per_frame) + 1
        next_frame = frame_count # the end
        if len(values) > 0:
            # set next_frame and next_v here to transition from starting seed
            time, next_v = values[0] 
            next_frame = frame = min(frame_count, time_to_frame(time, fps))
            if current_frame >= frame:
                _, v = values.pop(0)
                if len(values) > 0:
                    next_time, next_v = values[0]
                    next_frame = min(frame_count, time_to_frame(next_time, fps))
                else:
                    next_v = None
                    next_frame = frame_count
        steps = next_frame - current_frame
        if steps == 0:
            values_per_frame.append( value_fn(v, next_v, 1, steps) )
        else:
            for i in range(steps):
                values_per_frame.append( value_fn(v, next_v, i, steps) )
            
    #print(values_per_frame)
    return values_per_frame
