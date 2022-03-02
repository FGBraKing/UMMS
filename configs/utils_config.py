import yaml
from types import SimpleNamespace


def pretty_print_opt(in_opt):
    print(get_pretty_opt(in_opt))


def get_pretty_opt(in_opt):
    if isinstance(in_opt, dict):
        use_opt = in_opt
    else:
        use_opt = vars(in_opt)
    str_list = ['{:>35}: {:<45}'.format(k, repr(v)) for k, v in sorted(use_opt.items())]
    str_list.insert(0, '{:*^80s}'.format('Custom config'))
    str_list.append('{:*^80s}'.format('End'))
    message = '\n'.join(str_list)
    return message


def get_config(config):
    with open(config, 'r') as stream:
        loader = yaml.FullLoader(stream)
        out = loader.get_single_data()
        # out1 = yaml.load(stream)
        return out


def dict2obj(dic):
    return SimpleNamespace(**dic)


def obj2dict(obj):
    # return obj.__dict__
    return vars(obj)


# one_true_at_most
def only_one_true(*args):
    tmp = False
    for arg in args:
        if tmp:
            assert not arg, 'only one of {} can be true'.format(args)
        tmp = tmp or arg
    return tmp


class DictObj(object):
    def __init__(self, dic):
        self.dic = dic

    def __setattr__(self, key, value):
        if key == 'dic':
            object.__setattr__(self, key, value)
            return
        print('set attr called {},{}'.format(key, value))
        self.dic[key] = value

    def __getattr__(self, item):
        value = self.dic[item]
        if isinstance(value, dict):
            return DictObj(value)
        if isinstance(value, (list, tuple)):
            r = []
            for i in value:
                r.append(DictObj(i))
            return r
        else:
            return self.dic[item]

    def __getitem__(self, item):
        return self.dic[item]


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = ConfigDict(v)
                    if isinstance(v, list):
                        self.__convert(v)
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    v = ConfigDict(v)
                if isinstance(v, list):
                    # list: inplace
                    self.__convert(v)
                self[k] = v

    def __convert(self, v):
        '''
         列表还是列表， 列表里边的字典变成ConfigDict
        '''
        for elem in range(0, len(v)):
            if isinstance(v[elem], dict):
                v[elem] = ConfigDict(v[elem])
            elif isinstance(v[elem], list):
                self.__convert(v[elem])

    def __getattr__(self, item):
        return self.get(item)

    def __setitem__(self, key, value):
        super(ConfigDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delitem__(self, key):
        super(ConfigDict, self).__delitem__(key)
        del self.__dict__[key]

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, item):
        self.__delitem__(item)

    def __str__(self):
        # # old_version
        # message = ''
        # message += '----------------- Options ---------------\n'
        # for k, v in sorted(self.__dict__.items()):
        #     message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        # message += '----------------- End -------------------'

        str_list = ['{:>35}: {:<45}'.format(k, repr(v)) for k, v in sorted(self.__dict__.items())]
        str_list.insert(0, '{:*^80s}'.format('Custom config'))
        str_list.append('{:*^80s}'.format('End'))
        message = '\n'.join(str_list)
        # print(message)
        return message

    def merge_dict(self, **kwargs):
        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    v = ConfigDict(v)
                if isinstance(v, list):
                    self.__convert(v)
                self[k] = v
        # self.__dict__.update(kwargs)

    def merge_object(self, obj):
        if obj:
            for k, v in vars(obj).items():
                if isinstance(v, dict):
                    v = ConfigDict(v)
                if isinstance(v, list):
                    self.__convert(v)
                self[k] = v


def main():
    # tt = [{'hand': 15, 'ddd': 18, 'hg': {'fd': 22, 'love': 66}}, 4]
    tt = {'hand': [1, 2, 3], 'ddd': 18, 'hg': {'fd': 22, 'love': 66}}
    tt1 = {'hg1': {'fd1': 22, 'love1': [66, 77, 88]}}
    config = ConfigDict(tt)
    print(type(config))
    pretty_print_opt(config)


if __name__ == '__main__':
    main()
