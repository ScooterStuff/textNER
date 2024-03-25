import time
from memory_profiler import memory_usage
from motion_game_mapper import MotionGameMapper  # Import your class from the script where it's defined

def measure_execution_time_and_memory(func, *args, **kwargs):
    """
    Measures the execution time and memory usage of a function.
    
    :param func: The function to measure.
    :param args: Arguments to pass to the function.
    :param kwargs: Keyword arguments to pass to the function.
    :return: Execution time in seconds, peak memory usage in MiB.
    """
    start_time = time.time()  # Start time
    peak_memory = memory_usage((func, args, kwargs), max_usage=True)  # Peak memory usage
    end_time = time.time()  # End time
    
    execution_time = end_time - start_time  # Calculate execution time
    return execution_time, peak_memory

def main():
    mapper = MotionGameMapper()
    predict_text = "I want to play Minecraft with my right arm I want to jump when I pose thumb down I want to do index pinch to place down a block three fingers to destroy."
    output_file = "prediction_output.json"
    
    # Measure performance
    execution_time, peak_memory = measure_execution_time_and_memory(mapper.predict_to_json, predict_text, output_file)
    
    print(f"Execution Time: {execution_time:.4f} seconds")
    print(f"Peak Memory Usage: {peak_memory:.4f} MiB")

if __name__ == "__main__":
    main()
