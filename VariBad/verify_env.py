from environments.crypto_env import CryptoPortfolioEnv
import numpy as np

def verify_crypto_env():
    print("Initializing Environment...")
    env = CryptoPortfolioEnv(data_dir='data/1h') # Make sure data/1h exists and has files
    
    print("Resetting Environment...")
    obs = env.reset()
    print(f"Observation Shape: {obs.shape}")
    print(f"Initial Weights: {env.weights}")
    
    # Check random actions
    print("\nRunning Random Actions...")
    for i in range(5):
        action =  np.random.normal(size=env.action_space.shape) # Gaussian noise as logits
        obs, reward, done, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  Actions (Logits): {action[:3]}... (First 3)")
        print(f"  Weights: {info['portfolio_value']:.2f}")
        print(f"  Net Return: {info['return']:.4f}")
        print(f"  Cost: {info['cost']:.6f}")
        print(f"  Reward: {reward:.4f}")
        print("-" * 20)
        
        if done:
            break
            
    print("\nVerification Complete!")

if __name__ == "__main__":
    verify_crypto_env()
