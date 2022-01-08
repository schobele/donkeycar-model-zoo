import click
import os
import donkeycar as dk
import logging
import tarfile
import numpy as np
from donkeycar.parts.tub_v2 import Tub
from donkeycar.pipeline.types import TubRecord
from donkeycar.config import load_config


@click.command()
@click.option('--mycar', '-c')
@click.option('--tub', '-t')
def main(mycar, tub):
    AutofillConditions(mycar, tub)


class AutofillConditions(object):

    def __init__(self, mycar, tub):
        self.cfg = load_config(os.path.join(mycar, 'config.py'))

        if ' ' in tub:
            for tub in tub.split():
                self.makeCCmdTub(mycar, tub)
        else:
            self.makeCCmdTub(mycar, tub)

    def isStraight():
        return True

    def makeCCmdTub(self, my_car_path, tub):

        # put your path to donkey project
        tar = tarfile.open(os.path.expanduser(
            '../donkeycar/donkeycar/tests/tub/tub.tar.gz'))
        tub_parent = os.path.join(my_car_path, 'data/')
        new_tub_parent = os.path.join(my_car_path, 'data_ccmd/')
        tar.extractall(tub_parent)

        tub_path = os.path.join(tub_parent, tub)
        tub1 = Tub(tub_path)
        tub2 = Tub(os.path.join(new_tub_parent, tub),
                   inputs=['cam/image_array', 'user/angle', 'user/throttle',
                           "behavior/label", "behavior/one_hot_state_array", "behavior/state"],
                   types=['image_array', 'float', 'float', 'str', 'vector', 'int'])

        new_records = {}
        for key, record in enumerate(tub1):
            new_records[key] = record

        def next_records(id):
            if id + 10 < len(new_records):
                return [new_records[id+i] for i in range(10)]
            else:
                return False

        def getCurrentCCmd(angle, next_angles, turn_count):

            if angle > -0.8 and angle < 0.8 or turn_count < 5:
                left = all(iangle <= -0.6 for iangle in next_angles[4:9])
                right = all(iangle >= 0.6 for iangle in next_angles[4:9])
                if left:
                    return {"label": "Left", "state": 0, "one_hot_state_array": [1.0, 0.0, 0.0]}
                elif right:
                    return {"label": "Right", "state": 2, "one_hot_state_array": [0.0, 0.0, 1.0]}
                else:
                    return {"label": "Straight", "state": 1, "one_hot_state_array": [0.0, 1.0, 0.0]}
            else:
                return {"label": "Straight", "state": 1, "one_hot_state_array": [0.0, 1.0, 0.0]}

        turn_count = 0
        last_angle = 0
        for key, record in enumerate(tub1):

            t_record = TubRecord(config=self.cfg,
                                 base_path=tub1.base_path,
                                 underlying=record)
            img_arr = t_record.image(cached=False)
            record['cam/image_array'] = img_arr

            next10 = next_records(key)
            if next10:
                if '_s' in tub1:
                    ccmd = {"label": "Straight", "state": 1,
                            "one_hot_state_array": [0.0, 1.0, 0.0]}
                    record['behavior/label'] = ccmd["label"]
                    record['behavior/one_hot_state_array'] = ccmd["one_hot_state_array"]
                    record['behavior/state'] = ccmd["state"]
                else:
                    next10_angles = [rec['user/angle'] for rec in next10]
                    ccmd = getCurrentCCmd(
                        record['user/angle'], next10_angles, turn_count)
                    #record['user/conditional_command'] = ccmd

                    record['behavior/label'] = ccmd["label"]
                    record['behavior/one_hot_state_array'] = ccmd["one_hot_state_array"]
                    record['behavior/state'] = ccmd["state"]

                    if record['user/angle'] == last_angle:
                        turn_count += 1
                    else:
                        turn_count = 0

                    last_angle = record['user/angle']
                tub2.write_record(record)


if __name__ == "__main__":
    main()
