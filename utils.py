from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper

def generate_script(subject, video_length, creativity, api_key):
    title_temple = ChatPromptTemplate.from_messages(
        [
            ("huaman", "请为'{subject}'这个主题的视频想一个吸引人的标题")
        ]
    )
    script_temple = ChatPromptTemplate.from_messages(
        [
            ("huaman",
             """你是一位短视频的博主。根据以下标题和相关信息，为短视频频道写一个视频脚本。
             视频标题：{title}，视频时长：{duration}分钟，生成的脚本的长度尽量遵循视频时长的要求。
             要求开头抓住眼球，中间提供干货内容，结尾有惊喜。脚本格式也请按照【开头、中间、结尾】分隔。
             整体内容可以结合一下维基百科搜索出的信息，但仅供参考，只结合相关的即可，对不相关的进行忽略：
             '''{wikipedia_search}'''""")
        ]
    )

    model = ChatOpenAI(Open_api_key = api_key, temperature = creativity)

    title_chain = title_temple | model
    script_chain = script_temple | model

    title = title_chain.invoke({"subject": subject}).content

    search = WikipediaAPIWrapper(lang="zh")
    search_result = search.run(subject)

    script = script_chain.invoke({"title": title, "duration": video_length,
                                  "wikipedia_search": search_result}).content
    return title, script, search_result

