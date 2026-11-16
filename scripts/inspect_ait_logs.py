from collections import Counter
import re
from zipfile import ZipFile


def main():
    path = 'data/public/ait_lds/russellmitchell_no-pcaps.zip'
    names = ['gather/intranet_server/logs/auth.log.1', 'gather/intranet_server/logs/auth.log']
    pattern = re.compile(r'^([A-Z][a-z]{2} +[0-9]+ [0-9:]+) ([^ ]+) ([^:]+): (.*)$')
    programs, days, examples = Counter(), Counter(), []
    with ZipFile(path) as archive:
        for name in names:
            for line in archive.read(name).decode('utf-8').splitlines():
                match = pattern.match(line)
                if not match:
                    programs['UNPARSED'] += 1
                    continue
                program = re.sub(r'\[.*?\]', '', match.group(3))
                programs[program] += 1
                days[match.group(1)[:6]] += 1
                if program not in ('CRON', 'systemd-logind') and len(examples) < 20:
                    examples.append(line)
    print('Programs:', programs)
    print('Days:', days)
    print('Other examples:', *examples, sep='\n')


if __name__ == '__main__':
    main()
