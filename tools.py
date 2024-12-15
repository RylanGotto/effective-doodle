class BaseTools:
    async def invoke(self, name, query):
        return {
            name: getattr(self, name)
            for name, obj in self.__class__.__dict__.items()
            if callable(obj)
        }[name](query)


class WebSearch(BaseTools):
    def __init__(self, library):
        self.library = library

    def news(self, query):
        return self.library.news(query)

    def google(self, query):
        return self.library.google(query)
