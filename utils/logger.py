import logging


def get_logger(logfile):
    # 创建logger
    logger = logging.getLogger()
    # Log等级总开关
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入日志文件
    if not logfile:
        logfile = './log.txt'
    fh = logging.FileHandler(logfile, mode='a')
    # 用于写到file的等级开关
    fh.setLevel(logging.DEBUG)

    # 创建一个handler,用于输出到控制台
    ch = logging.StreamHandler()
    # 输出到console的log等级的开关
    ch.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


if __name__ == '__main__':
    logger = get_logger('')
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warning message')
    logger.error('error message')
    logger.critical('critical message')
