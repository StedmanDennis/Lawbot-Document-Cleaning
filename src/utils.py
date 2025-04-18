def print_progress_bar(iteration: int, total: int, length: int=50):
    percent = 100 * (iteration / total)
    filled_length = length * iteration // total
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r|{bar}| {percent:.1f}% Complete ({iteration} of {total})', end='', flush=True)
    if iteration == total:
        print()