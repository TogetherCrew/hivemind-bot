from .mongo import MongoSingleton


def fetch_community_id_by_guild_id(guild_id: str) -> str:
    """
    find the community id using the given guild id

    Parameters
    -----------
    guild_id : str
        the discord guild to find its community id

    Returns
    ---------
    community_id : str
        the community id that the guild is for
    """

    client = MongoSingleton.get_instance().get_client()

    platform = client["Core"]["platforms"].find_one(
        {"metadata.id": guild_id, "name": "discord"}, {"community": 1}
    )
    if platform is None:
        raise ValueError(f"The guild id  {guild_id} does not exist!")
    platform_id = str(platform["community"])
    return platform_id
