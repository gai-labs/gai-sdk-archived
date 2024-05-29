import React from "react";

function ColoredText({ children, badge }) {
    let colored = {
        backgroundColor: "#25c2a0",
        color: "#fff",
    };
    if (badge === "warning") {
        colored["backgroundColor"] = "#f5a623";
    }

    return (
        <span
            style={{
                ...colored,
                borderRadius: "5px",
                padding: "0.5rem",
            }}
        >
            {children}
        </span>
    );
}

export default ColoredText;
