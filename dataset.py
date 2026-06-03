import kagglehub

def dataset(handle: str) -> None:
    dataset_dir = kagglehub.competition_download(handle) 
    print(f'Dataset {handle} downloaded to {dataset_dir}')

dataset('spaceship-titanic')