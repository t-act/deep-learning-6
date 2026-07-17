from multiprocessing import Pool
from tqdm import tqdm

def square(x):
    return x**2

if __name__ == "__main__":
    numbers = list(range(10000))
    with Pool(processes=4) as pool:
        results = list(tqdm(
            pool.imap(square, numbers),
            total=len(numbers)
        ))