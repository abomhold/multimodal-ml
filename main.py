import pre_processing as pre


def main():
    mode_gender = pre.get_most_gender()
    print(mode_gender)

    mode_age = pre.get_most_age()
    print(mode_age)

    ope, con, ext, agr, neu = pre.get_avg_personality()
    print(f"Avg for ope: {ope:.3f}, con: {con:.3f}, ext: {ext:.3f}, agr: {agr:.3f}, neu: {neu:.3f}")


if __name__ == "__main__":
    main()
