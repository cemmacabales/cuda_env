#!/usr/bin/env python3
"""
Single Python script solving multiple tasks related to parallel reduction,
matrix operations, and CUDA kernel simulation using only the standard library
and numpy.
"""

import numpy as np

# ── Shared data definitions ───────────────────────────────────────────────
SALES = np.array([
    [12, 11, 4, 3],
    [12,  8, 5, 6],
    [ 3,  7, 8, 2]
], dtype=float)

FUELS = ["Diesel", "Gasoline", "Kerosene"]
DAYS  = ["Mon", "Tue", "Wed", "Thu"]

FIXED_PRICES = np.array([2, 1, 2], dtype=float)

VARYING_PRICES = np.array([
    [2, 3, 6, 7],
    [1, 8, 3, 7],
    [2, 3, 5, 1]
], dtype=float)


# ── Helper formatting functions ───────────────────────────────────────────
def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_matrix(mat, row_labels, col_labels, indent=0):
    prefix = " " * indent
    # Column headers
    print(prefix + f"{'':<12}", end="")
    for c in col_labels:
        print(f"{c:>10}", end="")
    print()
    print(prefix + "-" * (12 + 10 * len(col_labels)))
    # Rows
    for i, label in enumerate(row_labels):
        print(prefix + f"{label:<12}", end="")
        for j in range(mat.shape[1]):
            print(f"{mat[i, j]:>10.1f}", end="")
        print()


# ── Task 1 ────────────────────────────────────────────────────────────────
def parallel_reduction(arr):
    """
    Simulate parallel reduction by splitting the array in half,
    adding paired elements, and carrying over the last unpaired element
    when the length is odd.
    """
    current = list(arr)
    print(f"Initial array: {current}")
    step = 0

    while len(current) > 1:
        step += 1
        n = len(current)
        mid = n // 2
        left = current[:mid]
        right = current[mid:]
        active_threads = mid

        if n % 2 == 1:
            carry = current[-1]
            result = [left[i] + right[i] for i in range(mid)] + [carry]
        else:
            result = [left[i] + right[i] for i in range(mid)]

        print(f"  Step {step}:  Left={left}  |  Right={right}  "
              f"|  Threads={active_threads}  |  Result={result}")
        current = result

    print(f"Final sum: {current[0]}")
    return current[0]


def steps_needed(N):
    """Count reduction steps for an array of length N."""
    if N <= 0:
        return 0
    steps = 0
    current = N
    while current > 1:
        current = (current + 1) // 2
        steps += 1
    return steps


def task1():
    print_header("Task 1 – Parallel Reduction (Sum of Array)")
    arr = [2, 3, 3, 4, 4, 1, 6, 7, 4, 4]
    parallel_reduction(arr)

    print("\nTable of N vs t (steps needed) for N = 1 to 10:")
    print("-" * 30)
    print(f"{'N':<10} {'t (steps)':<10}")
    print("-" * 30)
    for N in range(1, 11):
        print(f"{N:<10} {steps_needed(N):<10}")


# ── Task 2A ───────────────────────────────────────────────────────────────
def task2a():
    print_header("Task 2A – Matrix Sales with Fixed Prices")
    revenue = SALES * FIXED_PRICES[:, np.newaxis]
    print("Revenue = Sales × Fixed Price per fuel\n")
    print_matrix(revenue, FUELS, DAYS)


# ── Task 2B ───────────────────────────────────────────────────────────────
def task2b():
    print_header("Task 2B – Matrix Sales with Varying Daily Prices")
    hadamard = SALES * VARYING_PRICES
    print("Revenue = Sales ⊙ Varying Prices (Hadamard product)\n")
    print_matrix(hadamard, FUELS, DAYS)
    return hadamard


# ── Task 2C ───────────────────────────────────────────────────────────────
def task2c(hadamard):
    print_header("Task 2C – Total Sales")
    per_fuel = np.sum(hadamard, axis=1)
    grand_total = np.sum(hadamard)

    print("Per-fuel totals:")
    for i, fuel in enumerate(FUELS):
        print(f"  {fuel:<12}: {per_fuel[i]:.1f}")
    print(f"\nGrand total: {grand_total:.1f}")


# ── Task 3 ────────────────────────────────────────────────────────────────
def compute_daily_sales(S, P):
    print("\n  ▸ Kernel: compute_daily_sales(S, P)")
    print("  " + "-" * 62)
    result = S * P
    print_matrix(result, FUELS, DAYS, indent=2)
    return result


def compute_total_sales(revenue, block_size):
    print("\n  ▸ Kernel: compute_total_sales(revenue, block_size={})".format(block_size))
    print("  " + "-" * 62)

    flat = revenue.flatten()
    n = flat.size
    total = 0.0

    print(f"  Flattened array: {flat.tolist()}")
    print(f"  Length: {n}, Block size: {block_size}\n")

    num_blocks = (n + block_size - 1) // block_size

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, n)
        block = flat[start:end].tolist()

        print(f"  Block {block_idx} (indices {start:>2}-{end - 1:>2}): {block}")

        # Shared-memory style reduction inside this block
        current = list(block)
        step = 0
        while len(current) > 1:
            step += 1
            m = len(current)
            mid = m // 2
            left = current[:mid]
            right = current[mid:]
            active = mid

            if m % 2 == 1:
                carry = current[-1]
                current = [left[i] + right[i] for i in range(mid)] + [carry]
            else:
                current = [left[i] + right[i] for i in range(mid)]

            print(f"    Step {step}: Left={left} | Right={right} | "
                  f"Threads={active} | Result={current}")

        partial = current[0] if current else 0.0
        print(f"    → Block {block_idx} partial sum: {partial}")
        total += partial
        print(f"    → Accumulated total after atomic-add: {total}\n")

    print(f"  Final accumulated total: {total}")
    return total


def task3():
    print_header("Task 3 – Simulate CUDA Kernels in Python")
    revenue = compute_daily_sales(SALES, VARYING_PRICES)
    compute_total_sales(revenue, block_size=4)


# ── Main entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    task1()
    task2a()
    hadamard = task2b()
    task2c(hadamard)
    task3()
    print("\n" + "=" * 70)
    print(" END OF REPORT")
    print("=" * 70 + "\n")
