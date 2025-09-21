import pickle
import random
import string
import argparse

def generate_random_string(min_len, max_len):
    """Generates a random string of lowercase letters and spaces."""
    length = random.randint(min_len, max_len)
    # Using letters and spaces can better simulate text structure than just random chars
    chars = string.ascii_lowercase + ' ' * 10  # Weight spaces more heavily
    return ''.join(random.choice(chars) for _ in range(length)).strip()

def create_debug_dataset(num_samples, min_len, max_len, output_path):
    """
    Creates a large dummy dataset with random string content for debugging purposes.
    This kind of data should result in a high, non-decreasing loss during training,
    which is useful for verifying that the training loop is functioning correctly.
    """
    print(f"Generating {num_samples} samples for the debug dataset...")
    dataset = []
    for i in range(num_samples):
        sample = {
            'system': generate_random_string(min_len, max_len),
            'user': generate_random_string(min_len, max_len),
            'response': generate_random_string(min_len, max_len)
        }
        dataset.append(sample)
    
    print(f"Saving dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    print("Debug dataset created successfully.")
    print(f"First generated sample: {dataset[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a large dummy dataset with random text for debugging SFT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of data samples to generate.')
    parser.add_argument('--min_len', type=int, default=50, help='Minimum length of random strings for each field.')
    parser.add_argument('--max_len', type=int, default=200, help='Maximum length of random strings for each field.')
    parser.add_argument('--output_path', type=str, default='debug_chat_traces.pkl', help='Path to save the output pickle file.')
    
    args = parser.parse_args()
    
    create_debug_dataset(
        num_samples=args.num_samples,
        min_len=args.min_len,
        max_len=args.max_len,
        output_path=args.output_path
    )