from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'lung_us'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files.
                (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mateo-drr',
    maintainer_email='mateodrr@gmail.com',
    description='Send Targets',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
		'talker = lung_us.publisher_member_function:main',
		'listener = lung_us.subscriber_member_function:main',
        ],
    },
)
