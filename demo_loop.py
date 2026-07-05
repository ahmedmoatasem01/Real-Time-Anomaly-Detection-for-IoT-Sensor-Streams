import time
import requests

API_URL = 'http://localhost:8000'

def run_loop():
    print('Starting demo loop... Will inject different faults every 15 seconds.')
    while True:
        # 1. Normal
        print('Stream: NORMAL')
        requests.post(f'{API_URL}/faults/stop')
        time.sleep(15)

        # 2. Gradual Drift (Low -> Medium)
        print('Stream: GRADUAL DRIFT (Low/Medium Severity)')
        requests.post(f'{API_URL}/faults/inject', json={'fault_type': 'gradual_drift', 'magnitude': 20.0, 'duration_steps': 100})
        time.sleep(15)

        # 3. Noise Burst (Medium Severity)
        print('Stream: NOISE BURST (Medium Severity)')
        requests.post(f'{API_URL}/faults/inject', json={'fault_type': 'noise_burst', 'magnitude': 5.0, 'duration_steps': 100})
        time.sleep(15)

        # 4. Spike (High/Critical Severity)
        print('Stream: SPIKE (High/Critical Severity)')
        requests.post(f'{API_URL}/faults/inject', json={'fault_type': 'spike', 'magnitude': 50.0, 'duration_steps': 100})
        time.sleep(15)

if __name__ == '__main__':
    run_loop()
