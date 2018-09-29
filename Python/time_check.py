import datetime


class CheckTime:
    def __init__(self):
        self.time = datetime.datetime.now()

    def start(self):
        self.time = datetime.datetime.now()

    def end(self):
        delta_time = datetime.datetime.now() - self.time
        if delta_time.days:
            print('{}d'.format(delta_time.days))

        seconds = delta_time.seconds + (delta_time.microseconds // 1000 / 1000)

        print('{}s'.format(seconds))
