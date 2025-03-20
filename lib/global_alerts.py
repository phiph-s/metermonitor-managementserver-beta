from threading import Lock

alerts = {}
alerts_lock = Lock()

def add_alert(key, alert):
    with alerts_lock:
        alerts[key] = alert

def remove_alert(key):
    with alerts_lock:
        if key in alerts:
            del alerts[key]

def get_alerts():
    with alerts_lock:
        return dict(alerts)

def clear_alerts():
    with alerts_lock:
        alerts.clear()
