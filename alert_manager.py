import os
import sys
import json
import numpy as np
from datetime import datetime
from fusion_engine import (predict_anomaly, predict_insect,
                            compute_risk_score, get_risk_level,
                            full_prediction, class_names)

print("=" * 55)
print("  FILE 7 of 8 — alert_manager.py")
print("  SMS alert notification system")
print("=" * 55)

# ── SMS Templates ─────────────────────────────────────────────────────
TEMPLATES = {
    'STABLE': (
        "\n[GRAIN STORE ALERT] STATUS: STABLE\n"
        "Time     : {ts}\n"
        "Temp     : {temp}°C\n"
        "Humidity : {hum}%RH\n"
        "CO Level : {co} ppm\n"
        "Insects  : {insect}\n"
        "Risk Score: {score}\n"
        "ACTION   : No action required. Continue routine monitoring."
    ),
    'MODERATE': (
        "\n[GRAIN STORE ALERT] STATUS: MODERATE RISK\n"
        "Time     : {ts}\n"
        "WARNING  : {param} is approaching limit\n"
        "Current  : {val}   Limit: {limit}\n"
        "Insect   : {insect} ({conf:.0%} confidence)\n"
        "Risk Score: {score}\n"
        "ACTION   : Inspect warehouse within 2 hours."
    ),
    'HIGH': (
        "\n[GRAIN STORE EMERGENCY] HIGH RISK DETECTED\n"
        "Time     : {ts}\n"
        "CRITICAL : {param} EXCEEDED THRESHOLD\n"
        "Current  : {val}   Limit: {limit}\n"
        "PEST     : {insect} ({conf:.0%} confidence)\n"
        "Risk Score: {score}\n"
        "ACTION   : IMMEDIATE ACTION REQUIRED. Call supervisor now."
    ),
}


class AlertManager:
    def __init__(self, phone_numbers,
                 use_twilio=False,
                 account_sid=None,
                 auth_token=None,
                 from_number=None):
        self.numbers       = phone_numbers
        self.current_level = None
        self.alert_log     = []
        self.use_twilio    = use_twilio

        if use_twilio:
            try:
                from twilio.rest import Client
                self.client  = Client(account_sid, auth_token)
                self.from_no = from_number
                print("  Twilio SMS enabled.")
            except ImportError:
                print("  Twilio not installed. Using print mode.")
                self.use_twilio = False

    def _determine_trigger(self, temp, hum, co):
        """Returns the triggering parameter details."""
        if temp > 50:
            return 'Temperature', f'{temp}°C',  '50°C'
        if hum  > 50:
            return 'Humidity',    f'{hum}%RH',  '50%RH'
        if co   > 25:
            return 'CO Level',    f'{co} ppm',  '25 ppm'
        return 'Environment', 'Multiple', 'Safe range'

    def _send(self, message, level):
        """Prints or sends the SMS message."""
        border = '─' * 58
        print(border)
        print(message)
        print(border)

        # Log the alert
        self.alert_log.append({
            'time'   : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level'  : level,
            'message': message.strip(),
        })

        # Send via Twilio if enabled
        if self.use_twilio:
            for num in self.numbers:
                try:
                    self.client.messages.create(
                        body    = message,
                        from_   = self.from_no,
                        to      = num
                    )
                    print(f"  SMS sent to {num}")
                except Exception as e:
                    print(f"  SMS failed: {e}")

    def evaluate(self, sensor_window_raw, image_path,
                 temp, hum, co, force=False):
        """
        Main evaluation function.
        Computes risk score and sends SMS if level changes.

        Parameters:
            sensor_window_raw : numpy (60, 3) raw sensor values
            image_path        : path to insect image
            temp, hum, co     : current raw sensor readings
            force             : True = always send SMS
        """
        ts = datetime.now().strftime('%d %b %Y %H:%M')

        # Get probabilities from both models
        result = full_prediction(
            sensor_window_raw, image_path, temp, hum, co
        )

        score  = result['risk_score']
        level  = result['risk_level']
        insect = result['insect']
        conf   = result['confidence']

        # Determine which parameter triggered the alert
        param, val, limit = self._determine_trigger(temp, hum, co)

        # Send SMS only when level changes or HIGH RISK persists
        should_send = (
            level != self.current_level or
            level == 'HIGH'             or
            force
        )

        if should_send:
            self.current_level = level
            msg = TEMPLATES[level].format(
                ts     = ts,
                temp   = temp,
                hum    = hum,
                co     = co,
                insect = insect,
                conf   = conf,
                param  = param,
                val    = val,
                limit  = limit,
                score  = score,
            )
            print(f"\nRisk Level: {level}")
            self._send(msg, level)
        else:
            print(f"  Risk: {score} ({level}) — no level change, "
                  f"SMS suppressed.")

        return result

    def print_log(self):
        """Prints all alerts that have been sent."""
        if not self.alert_log:
            print("No alerts logged yet.")
            return
        print(f"\n{'='*55}")
        print(f"  ALERT LOG ({len(self.alert_log)} alerts)")
        print(f"{'='*55}")
        for entry in self.alert_log:
            print(f"  [{entry['time']}] {entry['level']}")


# ── Test run ──────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Find a sample insect image for testing
    IMAGE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'insect_images'
    )

    sample_image = None
    for species in class_names:
        folder = os.path.join(IMAGE_DIR, species)
        if os.path.exists(folder):
            imgs = [f for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if imgs:
                sample_image = os.path.join(folder, imgs[0])
                break

    if sample_image is None:
        print("\nNo insect images found for testing.")
        print("Add images to data/insect_images/")
        sys.exit(0)

    print(f"\nUsing sample image: {sample_image}")

    # Create alert manager
    am = AlertManager(
        phone_numbers = ['+91XXXXXXXXXX'],
        use_twilio    = False   # Set True and add credentials to send real SMS
    )

    print("\n--- Test 1: STABLE condition ---")
    window_stable = np.random.rand(60, 3)
    window_stable[:, 0] = 30   # temp = 30°C (normal)
    window_stable[:, 1] = 40   # humidity = 40% (normal)
    window_stable[:, 2] = 10   # CO = 10 ppm (normal)
    r1 = am.evaluate(window_stable, sample_image,
                     temp=30, hum=40, co=10)

    print("\n--- Test 2: MODERATE condition ---")
    window_mod = np.random.rand(60, 3)
    window_mod[:, 1] = 0.8     # high humidity
    r2 = am.evaluate(window_mod, sample_image,
                     temp=38, hum=47, co=18)

    print("\n--- Test 3: HIGH RISK condition ---")
    window_high = np.random.rand(60, 3)
    window_high[:, 0] = 0.95   # very high temp
    window_high[:, 1] = 0.98   # very high humidity
    r3 = am.evaluate(window_high, sample_image,
                     temp=55, hum=53, co=28, force=True)

    am.print_log()

    print("\n✓ DONE — alert_manager.py complete")
    print("  Next: streamlit run app.py")
