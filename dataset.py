import kagglehub

def dataset(handle: str) -> None:
    try:
        dataset_dir = kagglehub.dataset_download(handle) 
        print(f'Dataset {handle} downloaded to {dataset_dir}')
    except ValueError:
        dataset_dir = kagglehub.competition_download(handle)
        print(f'Dataset {handle} downloaded to {dataset_dir}')
    except Exception as e:
        print(f'Error downloading dataset: {e}')


dataset('spaceship-titanic')