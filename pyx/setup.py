from setuptools import setup, find_packages

setup(
    name='pyx',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        cv2, datetime, inspect, kivy, pandas, pyautogui, reportlab, sqlite3, time
    ],
)
