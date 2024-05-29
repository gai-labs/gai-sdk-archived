import { themes } from "prism-react-renderer";

const lightCodeTheme = themes.palenight;
const darkCodeTheme = themes.dracula;

const organizationName = "gai-labs";
const projectName = "gai";
const config = {
    title: "Gai",
    tagline: "LLM Application Development Library",
    url: `https://${organizationName}.github.io`,
    baseUrl: `/${projectName}/`,
    onBrokenLinks: "throw",
    onBrokenMarkdownLinks: "warn",
    favicon: "img/favicon.ico",
    organizationName: organizationName,
    projectName: projectName,

    presets: [
        [
            "classic",
            {
                docs: {
                    sidebarPath: require.resolve("./sidebars.js"),
                    routeBasePath: "/",
                },
                theme: {
                    customCss: require.resolve("./src/css/custom.css"),
                },
            },
        ],
    ],

    themeConfig:
        /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
        ({
            colorMode: {
                disableSwitch: true,
            },
            navbar: {
                items: [
                    {
                        label: "Gai",
                        to: "/",
                        position: "left",
                    },
                ],
            },
            footer: {
                style: "dark",
                links: [
                    {
                        title: "Introduction",
                        items: [
                            {
                                label: "Introduction",
                                to: "/",
                            },
                        ],
                    },
                ],
                copyright: `Copyright © ${new Date().getFullYear()} GaiLabs Pte Ltd`,
            },
            prism: {
                theme: lightCodeTheme,
                darkTheme: darkCodeTheme,
                additionalLanguages: ["shell-session"],
            },
        }),
};

module.exports = config;
