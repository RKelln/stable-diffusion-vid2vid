
import unittest
from typing import Any

from vid2vid_schedules import *


def fix_seed(seed : Any) -> int:
    if seed is None:
        return -1
    
    s = int(seed)
    if s < 0:
        return -1

    return s


class TestTimeStringToSeconds(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(time_string_to_seconds(""), 0.0)
        self.assertEqual(time_string_to_seconds(" "), 0.0)
    
    def test_zero(self):
        self.assertEqual(time_string_to_seconds("0"), 0.0)
        self.assertEqual(time_string_to_seconds(" 0.0 "), 0.0)
    
    def test_seconds(self):
        self.assertEqual(time_string_to_seconds("1."), 1.0)
        self.assertEqual(time_string_to_seconds("1.0"), 1.0)
        self.assertEqual(time_string_to_seconds(" 1.00 "), 1.0)
        self.assertEqual(time_string_to_seconds("01.10"), 1.1)
        self.assertEqual(time_string_to_seconds("123425.6789"), 123425.679)

    def test_minutes(self):
        self.assertEqual(time_string_to_seconds("0:1."), 1.0)
        self.assertEqual(time_string_to_seconds("0:1"), 1.0)
        self.assertEqual(time_string_to_seconds(" 0:01 "), 1.0)
        self.assertEqual(time_string_to_seconds("00:01.0"), 1.0)
        self.assertEqual(time_string_to_seconds("00:01.00"), 1.0)
        self.assertEqual(time_string_to_seconds("00:01.10"), 1.1)
        self.assertEqual(time_string_to_seconds("12:34.6789"), 60*12 + 34.679)

    def test_minutes(self):
        self.assertEqual(time_string_to_seconds("0:1."), 1.0)
        self.assertEqual(time_string_to_seconds("0:1"), 1.0)
        self.assertEqual(time_string_to_seconds(" 0:01 "), 1.0)
        self.assertEqual(time_string_to_seconds("00:01.0"), 1.0)
        self.assertEqual(time_string_to_seconds("00:01.00"), 1.0)
        self.assertEqual(time_string_to_seconds("00:01.10"), 1.1)
        self.assertEqual(time_string_to_seconds("12:34.6789"), 12*60.0 + 34.679)
        self.assertEqual(time_string_to_seconds("59:59.999"), 59*60.0 + 59.999)
        self.assertEqual(time_string_to_seconds("59:80.999"), 59*60.0 + 80.999) # this is weird but allowed
        self.assertRaises(RuntimeError, time_string_to_seconds, "60:00.000")

    def test_hours(self):
        self.assertEqual(time_string_to_seconds("0:0:1."), 1.0)
        self.assertEqual(time_string_to_seconds(" 0:0:1 "), 1.0)
        self.assertEqual(time_string_to_seconds("00:00:01"), 1.0)
        self.assertEqual(time_string_to_seconds("1:0:0"), 3600.0)
        self.assertEqual(time_string_to_seconds("01:00:01.0"), 1.*3600.0 + 1.0)
        self.assertEqual(time_string_to_seconds("02:00:01.00"), 2.*3600.0 + 1.0)
        self.assertEqual(time_string_to_seconds("40:03:02.10"), 40.*3600.0 + 3.*60.0 + 2.1)
        self.assertEqual(time_string_to_seconds("1:12:34.6789"), 1.*3600.0 + 12*60.0 + 34.679)
        self.assertEqual(time_string_to_seconds("1:59:59.999"), 1.*3600.0 + 59*60.0 + 59.999)
        self.assertEqual(time_string_to_seconds("2:59:80.999"),2.*3600.0 + 59*60.0 + 80.999) # this is weird but allowed
        self.assertRaises(RuntimeError, time_string_to_seconds, "1:60:00.000")
        self.assertRaises(RuntimeError, time_string_to_seconds, "-1:0:0")


class TestParseToSeconds(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(parse_to_seconds("", 30.0, 10.0), 0.0)
        self.assertEqual(parse_to_seconds(" ", 30.0, 10.0), 0.0)
    
    def test_zero(self):
        self.assertEqual(parse_to_seconds("0", 30.0, 10.0), 0.0)

    def test_end_time(self):
        self.assertEqual(parse_to_seconds("1:00", 30.0, 10.0), 10.0)
        self.assertEqual(parse_to_seconds("10.", 30.0, 10.0), 10.0)
        self.assertEqual(parse_to_seconds("12.", 30.0, 11.0), 11.0)
        self.assertEqual(parse_to_seconds("13", 1.0, 10.0), 10.0) # frames @ 1 fps

    def test_frames(self):
        self.assertAlmostEqual(parse_to_seconds("0", 30.0, 1.0), 0.0, places=3)
        self.assertAlmostEqual(parse_to_seconds(" 1 ", 30.0, 1.0), 0.033, places=3)
        self.assertAlmostEqual(parse_to_seconds("1", 30.0, 10.0), 0.033, places=3)
        self.assertAlmostEqual(parse_to_seconds("10", 30.0, 10.0), 0.333, places=3)
        self.assertAlmostEqual(parse_to_seconds("5", 10.0, 10.0), 0.5, places=3)
        self.assertAlmostEqual(parse_to_seconds("999999", 10.0, 1.0), 1.0, places=3)

    def test_percent(self):
        self.assertAlmostEqual(parse_to_seconds("0%", 30.0, 1.0), 0., places=3)
        self.assertAlmostEqual(parse_to_seconds("1%", 30.0, 10.0), 0.1, places=3)
        self.assertAlmostEqual(parse_to_seconds("10%", 30.0, 10.0), 1., places=3)
        self.assertAlmostEqual(parse_to_seconds("05%", 10.0, 10.0), 0.5, places=3)
        self.assertAlmostEqual(parse_to_seconds("101%", 10.0, 1.0), 1.0, places=3)


class testTimeToFrame(unittest.TestCase):
    
    def test_zero(self):
        self.assertEqual(time_to_frame(0, 30.0), 1)
        self.assertEqual(time_to_frame(0.0, 30.0), 1)

    def test_frames(self):
        self.assertEqual(time_to_frame(0.1, 30.0), 4)
        self.assertEqual(time_to_frame(0.1, 10.0), 2)
        self.assertEqual(time_to_frame(0.3, 30.0), 10)
        self.assertEqual(time_to_frame(0.33, 30.0), 10)
        self.assertEqual(time_to_frame(0.333, 30.0), 10)
        self.assertEqual(time_to_frame(0.334, 30.0), 11)


class testFrameToTime(unittest.TestCase):
    
    def test_zero(self):
        self.assertEqual(frame_to_time(0, 10.0), 0.0)
        self.assertEqual(frame_to_time(0, 30.0), 0.0)

    def test_frames(self):
        self.assertEqual(frame_to_time(1, 10.0), 0.0)
        self.assertEqual(frame_to_time(2, 10.0), 0.1)
        self.assertEqual(frame_to_time(3, 30.0), 0.067)
        self.assertEqual(frame_to_time(4, 30.0), 0.1)
        self.assertEqual(frame_to_time(5, 30.0), 0.133)
        self.assertEqual(frame_to_time(7, 6.0), 1.0) # 6th frame last until the 1st second, 7th starts on 1.0


class TestParseSchedule(unittest.TestCase):
    def test_empty(self):
        default = 0
        self.assertEqual(parse_schedule("", 30.0, 10.0, default), [(0, default)])
        self.assertEqual(parse_schedule(" ", 30.0, 10.0, default), [(0, default)])
        default = 1
        self.assertEqual(parse_schedule("", 30.0, 10.0, default), [(0, default)])
        self.assertEqual(parse_schedule(" ", 30.0, 10.0, default), [(0, default)])
    
    def test_zero(self):
        default = -1
        self.assertEqual(parse_schedule("0:0", 30.0, 10.0, default), [(0, '0')])
        self.assertEqual(parse_schedule(" 0.:value ", 30.0, 10.0, default), [(0, 'value')])
    
    def test_time(self):
        default = -1
        self.assertEqual(parse_schedule("1:value ", 1.0, 5.0, default), [(parse_to_seconds("1", 1.0, 5.0), 'value')])
        self.assertEqual(parse_schedule("1:01:value ", 2.0, 6.0, default), [(parse_to_seconds("1:01", 2.0, 6.0), 'value')])
        self.assertEqual(parse_schedule("1:01:(value) ", 3.0, 7.0, default), [(parse_to_seconds("1:01", 3.0, 7.0), 'value')])
        self.assertEqual(parse_schedule("80%:(value) ", 4.0, 8.0, default), [(parse_to_seconds("80%", 4.0, 8.0), 'value')])

    def test_sequence(self):
        default = -1
        duration = 10.0
        fps = 30.0
        t1 = "1"
        t2 = "1.234"
        t3 = "80%"
        pt1 = parse_to_seconds(t1, fps, duration)
        pt2 = parse_to_seconds(t2, fps, duration)
        pt3 = parse_to_seconds(t3, fps, duration)

        self.assertEqual(parse_schedule(f"{t1}:v1,{t2}:v2,{t3}:v3", fps, duration, default), [(pt1, 'v1'), (pt2, "v2"), (pt3, "v3")])
        self.assertEqual(parse_schedule(f"{t3}:v3,{t1}:v1,{t2}:v2", fps, duration, default), [(pt1, 'v1'), (pt2, "v2"), (pt3, "v3")])
        self.assertEqual(parse_schedule(f"  {t3}:v3,  {t1}:v1, ,{t2}:v2  ,", fps, duration, default), [(pt1, 'v1'), (pt2, "v2"), (pt3, "v3")])


class TestTravelPlanning(unittest.TestCase):
#    travel_planning(values : list, frame_count : int, fps : float, value_fn: Callable[[Any, Any, int, int], Any], starting_value : any)

    def test_lerp(self):
        frame_count = 5
        fps = 20.0
        duration = float(frame_count) / fps
        starting_value = 0.
        values = [(0, 1.0), (parse_to_seconds("5", fps, duration), 2.0)]

        assertListAlmostEqual(self,
            travel_planning(values, frame_count, fps, basic_lerp, starting_value), 
            [1.0, 1.25, 1.5, 1.75, 2.0],
            places = 2)

        starting_value = 5.
        values = [(parse_to_seconds("2", fps, duration), 1.0), (parse_to_seconds("4", fps, duration), 2.0)]
        assertListAlmostEqual(self,
            travel_planning(values, frame_count, fps, basic_lerp, starting_value), 
            [5.0, 1.0, 1.5, 2.0, 2.0],
            places = 2)
 

def assertListAlmostEqual(self, first, second, places=None, context=None):
    """Asserts lists of lists or tuples to check if they compare and 
       shows which element is wrong when comparing two lists.
       Code from: https://stackoverflow.com/a/68851444
    """
    self.assertEqual(len(first), len(second), msg="List have different length")
    context = [first, second] if context is None else context
    for i in range(0, len(first)):
        if isinstance(first[0], tuple):
            context.append(i)
            self.assertListAlmostEqual(first[i], second[i], places, context=context)
        if isinstance(first[0], list):
            context.append(i)
            self.assertListAlmostEqual(first[i], second[i], places, context=context)
        elif isinstance(first[0], float):
            msg = "Difference in \n{} and \n{}\nFaulty element index={}".format(context[0], context[1], context[2:]+[i]) \
                if context is not None else None
            self.assertAlmostEqual(first[i], second[i], places, msg=msg)


if '__main__' == __name__:
    #unittest.main()

    fps = 30.0
    extract_fps = 5
    frame_count = 17
    duration = float(frame_count) / fps
    seed_schedule = ""
    seeds = [(t, fix_seed(s)) for t, s  in parse_schedule(seed_schedule, fps, duration, -1)]
    print("Seeds:", seeds == [(0,-1)])
    print(seeds)

    seed_travel = seed_travel_planning(seeds, frame_count, extract_fps, -1)
    print(seed_travel)
    

    seed_schedule = "0:1, 40%:2, 15:4"
    seeds = [(t, fix_seed(s)) for t, s  in parse_schedule(seed_schedule, fps, duration, -1)]
    print("Seeds:")
    print(seeds)
    seed_travel = seed_travel_planning(seeds, frame_count, extract_fps, 0)
    print(seed_travel)

    print("Denoise:")
    denoise_schedule = "3:0.1, 40%:0.2, 15:0.9"
    denoise = [(t, float(v)) for t, v in parse_schedule(denoise_schedule, fps, duration, 0.5)]
    print(denoise)
    noise_travel = denoise_travel_planning(denoise, frame_count, extract_fps, 0.3)
    print(noise_travel)

    print("Counts:")
    print(frame_count, len(seed_travel), len(noise_travel))

