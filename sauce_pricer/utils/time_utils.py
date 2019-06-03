

def get_time(starting_time, current_time):
    total_time = current_time - starting_time
    minutes = round(total_time // 60)
    seconds = round(total_time % 60)
    return '{} min., {} sec.'.format(minutes, seconds)