from __future__ import print_function, division

from ml_tools.dataset.facebook_links import FacebookLinks


class FacebookLinksUnique(FacebookLinks):
    builds = [
        {
            'target': 'facebook-links-unique.txt',
            'hash': {
                'value': 'e4daa2b6a3cb2761c3866a8dbe835f3d0d9a781a229d1d7743a7d7b3acfd7acd',
                'algorithm': 'sha256'
            }
        }
    ]

    @classmethod
    def post_process(cls):
        target_abspath = cls.abspath('facebook-links-unique.txt')
        org_abspath = FacebookLinks.abspath('facebook-links.txt')

        seen = set([])
        post_process_f = lambda line: tuple(
            sorted([int(v) for v in line.split()[:2]])
        )
        with open(org_abspath) as f_org:
            with open(target_abspath, 'w') as f_unique:
                for line in f_org:
                    res = post_process_f(line)
                    if res not in seen:
                        seen.add(res)
                        f_unique.write(line)


if __name__ == '__main__':
    FacebookLinksUnique.cmdline()
