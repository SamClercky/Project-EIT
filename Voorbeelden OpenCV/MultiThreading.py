"""
The threading library contains a number of objects which allow for concurrent-like behaviour of python code.
We say "concurrent-like" because Python utilizes a global interpreter lock to manage execution of code;
GIL allows only one instruction at a time by locking the interpreter which effectively makes Python single-threaded.

Normally this is not an issue and even has advantages such as shared memory. If true multithreading is needed
check ot the MultiProcessing library, which has a nearly identical API, although memory space is not shared.

Typical use cases threading include:
    Sensor polling at a set interval
    Long running processing tasks, ie. image processing
    Rest servers
"""

from threading import Thread, Lock
import time

'''
Import the threading library:
The Thread wraps any method as a background task which share memory within the main thread scope

The Lock allows for the synchronization of memory access between the main thread and the background threads
    Thread 1 will lock data['value'] so no other thread can access it while it remains locked
    Thread 1 will get the value of the data to manipulate: 0 <-- data['value']
    Thread 1 will increment the value of the memory 0 + 1 --> data['value'] == 1
    Thread 1 will release the lock, and allow Thread 2 to now lock data['value'] 
    Thread 2 will get the value of the data to manipulate: 1 <-- data['value']
    Thread 2 will decrement the value of the memory 1 - 1 --> data['value'] == 0
    
Try removing the 'with lock:' statement from the two functions below to observe a "race condition".
    Thread 1 will get the value of the data to manipulate: 0 <-- data['value']
    Thread 2 will get the value of the data to manipulate: 0 <-- data['value']
    Thread 1 will increment the value of the memory 0 + 1 --> data['value'] == 1
    Thread 2 will decrement the value of the memory 0 - 1 --> data['value'] == -1
    
'''

# A function which increments the data by 1 n times
def increment(n):
    print("The increment thread has started")
    for _ in range(n):
        # Lock the critical section of code to prevent race conditions
        with lock:
            data['value'] += 1
    print("The increment thread is complete")

# A function which decrements the data by 1 n times
def decrement(n):
    print("The decrement thread has started")
    for _ in range(n):
        # Lock the critical section of code to prevent race conditions
        with lock:
            data['value'] -= 1
    print("The decrement thread is complete")


# Initialize a dictionary with some data
data = {'value': 0}

# Create a lock object
lock = Lock()

# Number of iterations to do work
iterations = 20000000

# Create a thread to wrap the increment function
increment_thread = Thread(target=increment,
                          args=(iterations,),
                          daemon=True)

# Create a thread to wrap the increment function
decrement_thread = Thread(target=decrement,
                          args=(iterations, ),
                          daemon=True)

# Start the background threads
increment_thread.start()
decrement_thread.start()

# If you want to wait for the thread to complete before continuing, you can use the '.join()' method
# increment_thread.join()
# decrement_thread.join()

# Do some other work while the background tasks operate. In this example we just wait for the tasks to complete
t0 = time.monotonic()
while decrement_thread.is_alive() and increment_thread.is_alive():
    dt = time.monotonic() - t0
    print(f"\rBackground workers have been running for {dt:.2f} seconds")
    time.sleep(1.0)

print(f"All background workers are complete")
print(f"The final value in data is: {data['value']}")












