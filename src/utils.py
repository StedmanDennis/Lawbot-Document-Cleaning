def print_progress_bar(iteration: int, total: int, length: int=50):
    percent = f"{100 * (iteration / total):.1f}"
    filled_length = length * iteration // total
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r|{bar}| {percent}% Complete', end='', flush=True)