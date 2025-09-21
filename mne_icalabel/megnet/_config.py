MEGNET_NUMERICAL_TO_STRING: dict[int, str] = {
    0: "brain/other",
    1: "eye movement",
    2: "heart beat",
    3: "eye blink",
}

MEGNET_STRING_TO_NUMERICAL: dict[str, int] = {
    val: key for key, val in MEGNET_NUMERICAL_TO_STRING.items()
}
