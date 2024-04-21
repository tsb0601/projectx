import multiprocessing
import time

def worker_process(task_data):
    # Simulate some work
    print(f"Processing {task_data}")
    time.sleep(1)
    return f"Result for {task_data}"

def main():
    num_processes = 4  # Number of worker processes
    tasks = range(10)  # Dummy tasks
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            # map async allows non-blocking calls and better exception handling
            results = pool.map_async(worker_process, tasks)
            # get with timeout (optional) to fetch the results
            output = results.get(timeout=10)  # Adjust timeout to your needs
            print("Output:", output)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Proper cleanup: Terminate the pool
            pool.terminate()
            pool.join()
            print("Pool has been cleaned up.")

if __name__ == "__main__":
    main()

